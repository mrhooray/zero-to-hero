# nn-zero-to-hero

Following [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series

## Structure

| Directory | Topic |
|-----------|-------|
| `micrograd/` | Autograd engine and neural net library from scratch |
| `makemore/` | Character-level language model (bigram → MLP → WaveNet) |
| `gpt/` | GPT from scratch (bigram → attention masking → full GPT) |
| `gpt2/` | GPT-2 training with perf optimizations, DDP, and HellaSwag eval |

## Usage

```bash
uv run python <file>
uv run modal run gpt2/runner.py::train_ddp  # train on Modal GPU
uv run pytest  # micrograd tests
```
