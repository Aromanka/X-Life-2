"""
Unified LLM Interface
"""
import traceback
from typing import List, Dict, Optional, Any
from config_loader import get_config

# Import implementations
from core.llm.client import LLMClient, get_llm as get_api_llm
from core.llm.local_llm import LocalLLM, get_local_llm, is_local_mode
from core.llm.factory import get_llm_type, should_use_local


class UnifiedLLM:
    """
    Unified LLM wrapper that provides consistent interface
    for both local and API-based LLM modes.
    """

    def __init__(self):
        self._use_local = should_use_local()
        self._api_client: Optional[LLMClient] = None
        self._local_client: Optional[LocalLLM] = None
        self._fallback_to_api = False

    @property
    def llm_type(self) -> str:
        """Return current LLM type: 'local' or 'api'"""
        if self._fallback_to_api:
            return "api (fallback)"
        return "local" if self._use_local else "api"

    @property
    def is_local(self) -> bool:
        """Check if using local model"""
        return self._use_local and not self._fallback_to_api

    def _get_client(self):
        """Get the appropriate client, with fallback from local to API"""
        if self._use_local and not self._fallback_to_api:
            if self._local_client is None:
                try:
                    self._local_client = get_local_llm()
                    return self._local_client
                except Exception as e:
                    print(f"[WARN] Local model failed, falling back to API: {e}")
                    print(f"[DEBUG] {traceback.format_exc()}")
                    self._fallback_to_api = True

        if self._api_client is None:
            self._api_client = get_api_llm()
        return self._api_client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send chat request, return text content.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Generated text content
        """
        client = self._get_client()
        return client.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def chat_with_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        **kwargs
    ) -> dict:
        """
        Send chat request, return parsed JSON.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Parsed JSON as dict
        """
        client = self._get_client()
        return client.chat_with_json(messages, temperature=temperature, **kwargs)

    def extract_keywords(self, question: str, max_count: int = 3) -> List[str]:
        """
        Extract keywords from question.

        Args:
            question: Input question text
            max_count: Maximum number of keywords

        Returns:
            List of extracted keywords
        """
        client = self._get_client()
        return client.extract_keywords(question, max_count=max_count)

    def reload(self, force_local: Optional[bool] = None):
        """
        Reload the LLM (useful for switching modes)

        Args:
            force_local: If True, force local mode; if False, force API mode;
                         if None, use config
        """
        self._api_client = None
        self._local_client = None
        self._fallback_to_api = False

        if force_local is True:
            self._use_local = True
        elif force_local is False:
            self._use_local = False
        else:
            self._use_local = should_use_local()

        print(f"[INFO] UnifiedLLM reloaded, mode: {self.llm_type}")


# Global instance
_unified_llm: Optional[UnifiedLLM] = None


def get_unified_llm() -> UnifiedLLM:
    """Get the global unified LLM instance"""
    global _unified_llm
    if _unified_llm is None:
        _unified_llm = UnifiedLLM()
    return _unified_llm


# Convenience functions that match the existing LLMClient interface

def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """Convenience function for chat"""
    return get_unified_llm().chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)


def chat_with_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    **kwargs
) -> dict:
    """Convenience function for chat with JSON response"""
    return get_unified_llm().chat_with_json(messages, temperature=temperature, **kwargs)


def extract_keywords(question: str, max_count: int = 3) -> List[str]:
    """Convenience function for keyword extraction"""
    return get_unified_llm().extract_keywords(question, max_count=max_count)


# Backwards compatibility aliases
get_llm = get_unified_llm
get_llm_client = get_api_llm
