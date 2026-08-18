
#include <torch/extension.h>

void fused_qnorm_rope_kv_insert(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    torch::Tensor padded_q,
    torch::Tensor cache,
    const torch::Tensor& slots,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    double rms_eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_qnorm_rope_kv_insert", &fused_qnorm_rope_kv_insert);
}
