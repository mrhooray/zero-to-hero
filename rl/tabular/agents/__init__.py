from tabular.agents.monte_carlo_average import MonteCarloAverageAgent
from tabular.agents.monte_carlo_alpha import MonteCarloAlphaAgent
from tabular.agents.td_zero import TDZeroAgent
from tabular.agents.sarsa import SarsaAgent
from tabular.agents.sarsa_expected import SarsaExpectedAgent
from tabular.agents.q_learning import QLearningAgent
from tabular.agents.dyna_q import DynaQAgent

AGENTS = {
    agent.name: agent
    for agent in (
        MonteCarloAverageAgent,
        MonteCarloAlphaAgent,
        TDZeroAgent,
        SarsaAgent,
        SarsaExpectedAgent,
        QLearningAgent,
        DynaQAgent,
    )
}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
