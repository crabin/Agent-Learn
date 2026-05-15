"""Tiny local HelloAgents-compatible facade for examples in AgentEvaluation."""

from .agent import SimpleAgent
from .llm import HelloAgentsLLM

__all__ = ["HelloAgentsLLM", "SimpleAgent"]
