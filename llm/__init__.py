"""
llm — provider-agnostic LLM layer for sheares-policy-copilot.

Public surface:
    from llm.client import get_client      # LLMClient singleton
    from llm.settings import get_settings  # Settings singleton
    from llm.judge import GroundednessJudge
"""

from llm.client import LLMClient, get_client
from llm.settings import Settings, get_settings

__all__ = ["LLMClient", "get_client", "Settings", "get_settings"]
