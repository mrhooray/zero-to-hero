# Decode benchmark

The decode benchmark measures the complete OpenAI-compatible streaming path under rolling, closed-loop load. Its current workload uses an 8,192-token shared prefix, one uncached boundary token, and 1,024 generated tokens. Prefix reuse is an input condition that removes repeated prefill work; it is not the subject or name of the benchmark.

## Boundaries

- `tools/benchmark/benchmark_decode.py` starts a supplied server command, waits for
its HTTP endpoint, drives the benchmark, and records results.
- `tools/benchmark/decode.py` creates and validates the workload, applies
concurrency, checks streamed responses, and computes client-side metrics. It has no model, accelerator, kernel, container, or deployment imports.
- `tools/benchmark/profiles/` owns model selection, checkpoint identity, serving
arguments, image dependencies, accelerator topology, and prompt rendering.
- `tools/benchmark/launch_modal.py` selects a profile and provides Modal lifecycle
plumbing. It does not define the image or server command.

The profile creates the rendered workload at the start of each run using the checkpoint's codec. Generated inputs are validated before the server starts, so a stale checked-in prompt manifest cannot silently survive a tokenizer or source change.

## Workload

Four distinct prompt groups are generated. Groups diverge within their first 128 tokens, while requests within a group share exactly 8,192 rendered tokens. Each request adds one uncached token and asks for 1,024 output tokens with greedy sampling, EOS ignored, streaming enabled, and usage reporting enabled.

A fresh server receives two unmeasured concurrency-four priming waves. The first must report zero cached tokens and the second must report 8,192. For each requested concurrency, every persistent client slot then completes one warmup and 20 measured requests. A slot submits its successor immediately after the previous SSE response ends.

The result includes aggregate output tokens per second, requests per second, per-user decode speed, TTFT/TPOT/end-to-end percentiles, cache qualification, and rolling-occupancy evidence. Hardware-normalized metrics are intentionally not computed in the generic client; the selected profile records the topology needed to derive them.

## Running

Stage the pinned checkpoint once if its read-only Modal volume does not already exist:

```text
uv run python tools/checkpoint/launch_modal.py \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --volume <volume-name> \
  --create-volume
```

Then run the supported profile:

```text
uv run python tools/benchmark/launch_modal.py \
  --profile deepseek_v4_flash \
  --volume <volume-name> \
  --concurrency 4 16 64 128
```

The launcher writes one JSON result under `.artifacts/` unless `--output` is provided. Modal execution requires explicit authorization and four B200 GPUs; the CPU test suite only validates workload construction, boundary ownership, process lifecycle, and result plumbing.

Both DeepSeek V4 Flash and GLM 5.3 Flash have deployment profiles under `tools/benchmark/profiles/`; the generic benchmark modules stay unchanged.
