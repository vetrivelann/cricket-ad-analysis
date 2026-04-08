"""
Gemini LLM client — thin wrapper that delegates to gemini_llm.py.

Every other file in the project imports from this module
(gemini_client.ask_gemini, gemini_client.is_available, etc.).
This wrapper ensures backward compatibility: nothing else needs to change.
"""
from gemini_llm import (          # noqa: F401
    ask_gemini,
    ask_gemini_json,
    is_available,
    reset,
    test_connection,
)
