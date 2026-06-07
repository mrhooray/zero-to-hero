from tabular.agents.monte_carlo_average import MonteCarloAverageAgent
from tabular.agents.monte_carlo_alpha import MonteCarloAlphaAgent
from tabular.agents.td_zero import TDZeroAgent
from tabular.agents.sarsa import SarsaAgent

AGENTS = {
    agent.name: agent
    for agent in (
        MonteCarloAverageAgent,
        MonteCarloAlphaAgent,
        TDZeroAgent,
        SarsaAgent,
    )
}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
