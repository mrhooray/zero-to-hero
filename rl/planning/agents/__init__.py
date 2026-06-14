from planning.agents.vi import ValueIterationAgent

AGENTS = {agent.name: agent for agent in (ValueIterationAgent,)}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
