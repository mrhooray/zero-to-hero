# mojo-gpu-puzzles

Following [Mojo GPU Puzzles](https://puzzles.modular.com/)

## Local

```bash
pixi run p01           # run a puzzle
```

## Modal

```bash
PUZZLE=p01 modal run modal_run.py                # run a puzzle (default GPU: A10)
PUZZLE=p01 GPU=A100-80GB modal run modal_run.py  # choose GPU tier
```

See [Modal GPU docs](https://modal.com/docs/guide/gpu) for available GPU types.

## Debug

```bash
modal shell modal_debug.py::debug
```
