from deep.agents.reinforce import ReinforceAgent

AGENTS = {agent.name: agent for agent in (ReinforceAgent,)}

AGENT_NAMES = tuple(AGENTS)

__all__ = [
    "AGENTS",
    "AGENT_NAMES",
]
