# mojo-gpu-puzzles

Following [Mojo GPU Puzzles](https://puzzles.modular.com/)

## Local

```bash
pixi run p01           # run a puzzle
```

## Modal

```bash
modal run modal_run.py --puzzle p01                                                        # run a puzzle (default GPU: A10)
modal run modal_run.py --puzzle p01 --gpu A100-80GB                                        # choose GPU tier
modal run modal_run.py --puzzle p09 --args "--first-case"                                  # pass args to puzzle
modal run modal_run.py --puzzle p10 --sanitizer --tool memcheck --args "--memory-bug"      # run with compute-sanitizer
modal run modal_run.py --puzzle p10 --sanitizer --tool racecheck --args "--race-condition"
```

See [Modal GPU docs](https://modal.com/docs/guide/gpu) for available GPU types.

## Debug

```bash
modal shell modal_debug.py::debug
```

## Puzzles

| Puzzle | Topic |
| --- | --- |
| p01 | 1D map kernel and raw thread indexing |
| p02 | Zip two inputs with elementwise addition |
| p03 | Bounds guards for memory-safe kernels |
| p04 | 2D map with raw memory and TileTensor |
| p05 | Broadcast scalar/vector values across tiles |
| p06 | Multi-block 1D indexing |
| p07 | 2D block and thread indexing |
| p08 | Shared memory tile staging and barriers |
| p09 | GPU debugging workflow and failing cases |
| p10 | Compute-sanitizer memcheck and racecheck |
| p11 | Pooling with shared-memory reduction |
| p12 | Dot product via parallel reduction |
| p13 | 1D convolution, including block boundaries |
| p14 | Prefix sum scan across block phases |
| p15 | Axis-wise matrix reduction |
| p16 | Matmul from naive to tiled shared memory |
| p17 | MAX Graph custom 1D convolution op |
| p18 | MAX Graph custom softmax op |
| p19 | Attention op with transpose, softmax, matmul |
| p20 | PyTorch custom 1D convolution op |
| p21 | Embedding op and coalesced access |
| p22 | Fused layernorm-linear with backward pass |
| p23 | Functional patterns, tiling, SIMD vectorization |
| p24 | Warp fundamentals and warp-level reductions |
| p25 | Warp shuffle-down and broadcast communication |
| p26 | Warp shuffle-xor and prefix-sum patterns |
| p27 | Block sum, prefix sum, and broadcast |
| p28 | Async memory copies and overlap |
| p29 | Barriers, mbarriers, and double buffering |
| p30 | GPU profiling and cache behavior |
| p31 | Occupancy and resource-use tradeoffs |
| p32 | Shared-memory bank conflicts |
| p33 | Tensor core matrix multiplication |
| p34 | Cluster coordination, reductions, and hierarchy |
