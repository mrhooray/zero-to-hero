from tabular.agents.monte_carlo_average import MonteCarloAverageAgent
from tabular.agents.monte_carlo_alpha import MonteCarloAlphaAgent

AGENTS = {
    agent.name: agent
    for agent in (
        MonteCarloAverageAgent,
        MonteCarloAlphaAgent,
    )
}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
