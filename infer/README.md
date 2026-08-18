# Inference Zero to Hero

WIP LLM inference engine inspired by nano-vLLM, mini-SGLang, and TokenSpeed

## Preliminary results

![DeepSeek V4 Flash decode performance](assets/deepseek-v4-flash-decode.svg)

![GLM 5.3 Flash decode performance](assets/glm53-flash-decode.svg)

## CLI

### Testing

Run the local test suite:

```bash
uv run pytest -q tests --ignore=tests/gpu
```

Run the GPU tests on Modal:

```bash
INFER_MODAL_GPU=B200 uv run modal run -e <env> tools/modal_pytest.py --selector tests/gpu
```

### Checkpoint

Stage both pinned checkpoints into separate directories of one Modal Volume:

```bash
uv run python tools/checkpoint/launch_modal.py \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --volume <volume> \
  --volume-subpath deepseek-v4-flash-0731-7872f01b \
  --create-volume \
  -e <env>

uv run python tools/checkpoint/launch_modal.py \
  --model zai-org/GLM-5.3-Flash \
  --volume <volume> \
  --volume-subpath glm-5.3-flash-04c4e9e \
  -e <env>
```

### Serving

Start Infer on a prepared four-GPU host:

```bash
uv run --extra gpu infer serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --checkpoint-dir <checkpoint> \
  --parallelism dep4 \
  --speculation native \
  --host 0.0.0.0 \
  --port 8000

uv run --extra gpu infer serve zai-org/GLM-5.3-Flash \
  --checkpoint-dir <checkpoint> \
  --parallelism dep4 \
  --speculation native \
  --host 0.0.0.0 \
  --port 8000
```

`--speculation native` enables DSpark for DeepSeek V4 Flash 0731 or NextN for
GLM 5.3 Flash; use `none` for target-only decoding.

### Output parity

Compare deterministic output against another OpenAI-compatible engine:

```bash
uv run python tools/output_parity.py \
  --candidate-endpoint <candidate-url> \
  --reference-endpoint <reference-url> \
  --candidate-model <model> \
  --reference-model <model> \
  --tokenizer <checkpoint>/tokenizer.json \
  --output .artifacts/output_parity.json
```

Both candidate and reference servers must already be running.

### Benchmark

Run an Infer concurrency sweep:

```bash
uv run python tools/benchmark/launch_modal.py \
  --profile deepseek_v4_flash \
  --concurrency 4 16 64 128 \
  --speculation <native|none> \
  --volume <volume> \
  --volume-subpath deepseek-v4-flash-0731-7872f01b \
  -e <env>
```

- `native`: DSpark for DeepSeek V4 Flash 0731; NextN for GLM 5.3 Flash.
- `none`: target-only Infer; required by TokenSpeed and SGLang.
- External engines: add `--engine tokenspeed` or `--engine sglang`.
- GLM 5.3 Flash: use `--profile glm53_flash` and
  `--volume-subpath glm-5.3-flash-04c4e9e`.
