"""Backward-compatible conversational-agent entry point."""
from app.agent.factory import create_conversational_agent

# Preserve the original public name while keeping the implementation in app/agent/.
conversational_agent = create_conversational_agent()

__all__ = ["conversational_agent", "create_conversational_agent"]
