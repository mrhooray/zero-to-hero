from planning.agents.vi import ValueIterationAgent
from planning.agents.pi import PolicyIterationAgent

AGENTS = {agent.name: agent for agent in (ValueIterationAgent, PolicyIterationAgent)}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
