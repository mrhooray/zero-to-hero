from tabular.agents.monte_carlo_average import MonteCarloAverageAgent
from tabular.agents.monte_carlo_alpha import MonteCarloAlphaAgent
from tabular.agents.td_zero import TDZeroAgent
from tabular.agents.sarsa import SarsaAgent
from tabular.agents.sarsa_expected import SarsaExpectedAgent

AGENTS = {
    agent.name: agent
    for agent in (
        MonteCarloAverageAgent,
        MonteCarloAlphaAgent,
        TDZeroAgent,
        SarsaAgent,
        SarsaExpectedAgent,
    )
}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
