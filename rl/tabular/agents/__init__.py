from tabular.agents.monte_carlo_average import MonteCarloAverageAgent
from tabular.agents.monte_carlo_alpha import MonteCarloAlphaAgent
from tabular.agents.td_zero import TDZeroAgent

AGENTS = {
    agent.name: agent
    for agent in (
        MonteCarloAverageAgent,
        MonteCarloAlphaAgent,
        TDZeroAgent,
    )
}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
