from deep.agents.reinforce import ReinforceAgent
from deep.agents.dqn import DQNAgent
from deep.agents.ppo import PPOAgent
from deep.agents.sac import SACAgent

AGENTS = {
    agent.name: agent
    for agent in (
        ReinforceAgent,
        DQNAgent,
        PPOAgent,
        SACAgent,
    )
}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
