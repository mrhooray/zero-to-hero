from planning.agents.vi import ValueIterationAgent
from planning.agents.pi import PolicyIterationAgent
from planning.agents.mcts import MCTSAgent

AGENTS = {
    agent.name: agent
    for agent in (ValueIterationAgent, PolicyIterationAgent, MCTSAgent)
}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
