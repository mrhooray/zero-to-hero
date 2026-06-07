from tabular.agents.monte_carlo_average import MonteCarloAverageAgent

AGENTS = {agent.name: agent for agent in (MonteCarloAverageAgent,)}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
