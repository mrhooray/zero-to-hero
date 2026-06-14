# RL Zero To Hero

## Structure

| Track | Description |
| --- | --- |
| `tabular/` | Classic discrete-state RL on GridWorld |
| `deep/` | Neural RL for environments like CartPole and beyond |

## Learning Methods

| Path | Summary |
| --- | --- |
| `tabular/agents/mc-average` | First-visit / every-visit Monte Carlo with running averages |
| `tabular/agents/mc-alpha` | Monte Carlo control with constant learning rate |
| `tabular/agents/td-zero` | State-value TD(0) prediction |
| `tabular/agents/sarsa` | On-policy TD control with delayed bootstrap |
| `tabular/agents/sarsa-expected` | Expected SARSA with epsilon-greedy expectation |
| `tabular/agents/q-learning` | Off-policy TD control with greedy bootstrap |
| `tabular/agents/dyna-q` | Model-based Dyna-Q with planning steps |

## CLI

```bash
uv run tabular/cli.py render --size 8 --seed 42
uv run tabular/cli.py train --algorithm q-learning --train-episodes 512 --show-policy --plot tabular/plot/q_learning.png
uv run tabular/cli.py benchmark --train-episodes 512 --eval-episodes 32 --plot tabular/plot/benchmark.png
```

## Test

```bash
uv run pytest
```
