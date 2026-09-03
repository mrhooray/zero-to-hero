
// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Reduced from TokenSpeed's DeepSeek V4 writer at f17b03efc1728875c586d848f49da5905032e87c.

#include <cmath>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 512;
constexpr int kTPHeads = 16;
constexpr int kOutputHeads = 64;
constexpr int kPageSize = 64;
constexpr int kRopeDim = 64;
constexpr int kHalfRopeDim = 32;
constexpr int kNopeDim = 448;
constexpr int kTokenDataBytes = 576;
constexpr int kScaleBytes = 8;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kElemsPerLane = 16;
constexpr float kFp8Max = 448.0f;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(kWarpMask, value, mask, kWarpSize);
  }
  return value;
}

__device__ __forceinline__ float four_lane_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(kWarpMask, value, 1));
  return fmaxf(value, __shfl_xor_sync(kWarpMask, value, 2));
}

__global__ void writer_kernel(
    const nv_bfloat16* __restrict__ q,
    const nv_bfloat16* __restrict__ kv,
    nv_bfloat16* __restrict__ padded_q,
    uint8_t* __restrict__ cache,
    const int32_t* __restrict__ slots,
    const int32_t* __restrict__ positions,
    const float* __restrict__ cos_sin,
    float rms_eps,
    int tokens,
    int heads,
    int64_t page_stride,
    int64_t max_slots) {
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp = blockIdx.x * warps_per_block + threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int tasks_per_token = heads + 1;
  const int token = warp / tasks_per_token;
  const int task = warp % tasks_per_token;
  if (token >= tokens) {
    return;
  }

  const bool is_kv = task == heads;
  const int dim_base = lane * kElemsPerLane;
  const nv_bfloat16* source = is_kv
      ? kv + static_cast<int64_t>(token) * kHeadDim + dim_base
      : q + (static_cast<int64_t>(token) * heads + task) * kHeadDim + dim_base;
  const uint4 input0 = *reinterpret_cast<const uint4*>(source);
  const uint4 input1 = *reinterpret_cast<const uint4*>(source + 8);
  const nv_bfloat16* input0_bf16 = reinterpret_cast<const nv_bfloat16*>(&input0);
  const nv_bfloat16* input1_bf16 = reinterpret_cast<const nv_bfloat16*>(&input1);
  float values[kElemsPerLane];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] = __bfloat162float(input0_bf16[i]);
    values[i + 8] = __bfloat162float(input1_bf16[i]);
  }

  if (!is_kv) {
    float sum_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      sum_squares += values[i] * values[i];
    }
    const float scale = rsqrtf(warp_sum(sum_squares) / kHeadDim + rms_eps);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      values[i] *= scale;
    }
  }

  const bool rope_lane = dim_base >= kNopeDim;
  if (rope_lane) {
    const float* cos = cos_sin + positions[token] * kRopeDim;
    const float* sin = cos + kHalfRopeDim;
    const int rope_base = dim_base - kNopeDim;
#pragma unroll
    for (int pair = 0; pair < kElemsPerLane / 2; ++pair) {
      const int rope_pair = rope_base / 2 + pair;
      const float even = values[2 * pair];
      const float odd = values[2 * pair + 1];
      values[2 * pair] = even * cos[rope_pair] - odd * sin[rope_pair];
      values[2 * pair + 1] = even * sin[rope_pair] + odd * cos[rope_pair];
    }
  }

  if (!is_kv) {
    uint4 output0;
    uint4 output1;
    nv_bfloat16* output0_bf16 = reinterpret_cast<nv_bfloat16*>(&output0);
    nv_bfloat16* output1_bf16 = reinterpret_cast<nv_bfloat16*>(&output1);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      output0_bf16[i] = __float2bfloat16(values[i]);
      output1_bf16[i] = __float2bfloat16(values[i + 8]);
    }
    nv_bfloat16* destination =
        padded_q +
        (static_cast<int64_t>(token) * kOutputHeads + task) * kHeadDim +
        dim_base;
    *reinterpret_cast<uint4*>(destination) = output0;
    *reinterpret_cast<uint4*>(destination + 8) = output1;
    return;
  }

  const int64_t slot = slots[token];
  if (slot < 0 || slot >= max_slots) {
    return;
  }
  const int64_t page = slot / kPageSize;
  const int64_t row = slot % kPageSize;
  uint8_t* page_base = cache + page * page_stride;
  uint8_t* token_data = page_base + row * kTokenDataBytes;
  uint8_t* token_scales =
      page_base + static_cast<int64_t>(kPageSize) * kTokenDataBytes + row * kScaleBytes;

#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    values[i] = __bfloat162float(__float2bfloat16(values[i]));
  }

  if (!rope_lane) {
    float local_absmax = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      local_absmax = fmaxf(local_absmax, fabsf(values[i]));
    }
    const float absmax = fmaxf(four_lane_max(local_absmax), 1.0e-4f);
    int exponent = ilogbf(absmax) - 8;
    exponent += absmax > ldexpf(kFp8Max, exponent);
    const float inverse_scale = exp2f(-exponent);
    uint4 output;
    uint8_t* output_bytes = reinterpret_cast<uint8_t*>(&output);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      const float scaled = fminf(fmaxf(values[i] * inverse_scale, -kFp8Max), kFp8Max);
      output_bytes[i] = static_cast<uint8_t>(
          __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
    }
    *reinterpret_cast<uint4*>(token_data + dim_base) = output;
    if ((lane & 3) == 0) {
      token_scales[lane >> 2] = static_cast<uint8_t>(
          fminf(fmaxf(static_cast<float>(exponent) + 127.0f, 0.0f), 255.0f));
    }
    if (lane == 0) {
      token_scales[7] = 0;
    }
    return;
  }

  uint4 output0;
  uint4 output1;
  nv_bfloat16* output0_bf16 = reinterpret_cast<nv_bfloat16*>(&output0);
  nv_bfloat16* output1_bf16 = reinterpret_cast<nv_bfloat16*>(&output1);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    output0_bf16[i] = __float2bfloat16(values[i]);
    output1_bf16[i] = __float2bfloat16(values[i + 8]);
  }
  nv_bfloat16* rope =
      reinterpret_cast<nv_bfloat16*>(token_data + kNopeDim) + dim_base - kNopeDim;
  *reinterpret_cast<uint4*>(rope) = output0;
  *reinterpret_cast<uint4*>(rope + 8) = output1;
}

}  // namespace

void fused_qnorm_rope_kv_insert(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    torch::Tensor padded_q,
    torch::Tensor cache,
    const torch::Tensor& slots,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    double rms_eps) {
  TORCH_CHECK(
      q.is_cuda() && kv.is_cuda() && padded_q.is_cuda() && cache.is_cuda());
  TORCH_CHECK(slots.is_cuda() && positions.is_cuda() && cos_sin.is_cuda());
  TORCH_CHECK(q.scalar_type() == at::kBFloat16);
  TORCH_CHECK(kv.scalar_type() == at::kBFloat16);
  TORCH_CHECK(padded_q.scalar_type() == at::kBFloat16);
  TORCH_CHECK(cache.scalar_type() == at::kByte);
  TORCH_CHECK(slots.scalar_type() == at::kInt);
  TORCH_CHECK(positions.scalar_type() == at::kInt);
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat);
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && padded_q.is_contiguous());
  TORCH_CHECK(slots.is_contiguous() && positions.is_contiguous());
  TORCH_CHECK(cos_sin.is_contiguous() && cache.stride(1) == 1);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kHeadDim);
  TORCH_CHECK(q.size(1) == kTPHeads || q.size(1) == kOutputHeads);
  TORCH_CHECK(padded_q.dim() == 4 && padded_q.size(0) == q.size(0));
  TORCH_CHECK(padded_q.size(1) == 1 && padded_q.size(2) == kOutputHeads);
  TORCH_CHECK(padded_q.size(3) == kHeadDim);
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == kHeadDim);
  TORCH_CHECK(
      cache.dim() == 2 &&
      cache.size(1) >= kPageSize * (kTokenDataBytes + kScaleBytes));
  TORCH_CHECK(slots.dim() == 1 && slots.size(0) == q.size(0));
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == q.size(0));
  TORCH_CHECK(kv.size(0) == q.size(0));
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == kRopeDim);
  TORCH_CHECK(kv.device() == q.device() && padded_q.device() == q.device());
  TORCH_CHECK(cache.device() == q.device() && slots.device() == q.device());
  TORCH_CHECK(positions.device() == q.device() && cos_sin.device() == q.device());

  c10::cuda::CUDAGuard guard(q.device());
  const int tokens = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int total_warps = tokens * (heads + 1);
  constexpr int warps_per_block = kThreads / kWarpSize;
  const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  writer_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const nv_bfloat16*>(kv.data_ptr<at::BFloat16>()),
      reinterpret_cast<nv_bfloat16*>(padded_q.data_ptr<at::BFloat16>()),
      cache.data_ptr<uint8_t>(),
      slots.data_ptr<int32_t>(),
      positions.data_ptr<int32_t>(),
      cos_sin.data_ptr<float>(),
      static_cast<float>(rms_eps),
      tokens,
      heads,
      cache.stride(0),
      cache.size(0) * kPageSize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
