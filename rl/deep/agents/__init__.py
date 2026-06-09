from deep.agents.reinforce import ReinforceAgent
from deep.agents.dqn import DQNAgent

AGENTS = {
    agent.name: agent
    for agent in (
        ReinforceAgent,
        DQNAgent,
    )
}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
