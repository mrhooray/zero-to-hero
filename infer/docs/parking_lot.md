# Parking lot

These items need B200 builds, profiling, or quality evidence. Current behavior is preserved until that evidence exists.

## DeepSeek state precision

`models/deepseek_v4_flash/model.py` marks the packed SWA and compressed-history layouts as candidates. Validate cache precision against the pinned checkpoint and the end-to-end quality gates before changing their dtypes, packing, or byte layout.

## DeepSeek capacity sizing

`models/deepseek_v4_flash/serve.py` still uses provisional live-slot and history capacity constants. Replace them only with a startup memory plan measured against the complete resident model, graph pools, workspaces, and prefix snapshots on four B200 GPUs.

## GLM FlashKDA build provenance

The GLM prefill operation imports the external `flash_kda_C` extension, whose pinned installer lives at `tools/kernels/install_flash_kda.py` (commits, hashes, and patch identity match `prefill_kda.LICENSE.txt`). The installer still needs to be built and exercised with the production CUDA toolchain before the provenance notice counts as qualified.

## GLM decode benchmark profile

The decode harness is model-agnostic, and both models now have deployment profiles in `tools/benchmark/profiles/` without changing the generic workload or measurement modules. The GLM profile still needs independent B200 baseline results.

## Four-B200 host inventory

Kernel and topology validation must happen on the exact four-B200 host and its NVSwitch partition before choosing `dep4` as default.

## Decode workload normalization

`tools/benchmark/decode.py` embeds `messages_sha256`, `prompt_token_sha256`, and `suffix_token_sha256` in each of the 4,460 requests even though only four prefix groups exist, so `build_workload` and `validate_workload` currently memoize per-group digests and renders. Storing receipts once per group with requests carrying only spec plus group id would remove the caches and shrink the workload, but it changes `WORKLOAD_SCHEMA` and the `benchmark_decode` runner plus per-request self-containment, so keep the denormalized receipts until a workload revision is planned.
