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
