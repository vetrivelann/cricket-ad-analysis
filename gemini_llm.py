"""
Gemini LLM module for the Cricket Ad Analytics system.
Uses the new google-genai SDK (replaces deprecated google-generativeai).

This is a standalone module — no other files need to change.
All existing code that imports from gemini_client.py continues to work
because gemini_client.py now delegates to this module.
"""
import os
import json
import time
import logging
from dotenv import load_dotenv

# Always load .env so the API key is available regardless of import order
load_dotenv()

log = logging.getLogger(__name__)

_client = None
_init_attempted = False
_last_call_time = 0.0
_MIN_INTERVAL = 4.0        # seconds between calls — free tier is 15 RPM
_MAX_RETRIES = 3
_MODEL_NAME = None


def _get_client():
    """Initialize the google-genai Client. Called once and cached."""
    global _client, _init_attempted, _MODEL_NAME

    if _client is not None:
        return _client

    if _init_attempted:
        return None

    _init_attempted = True

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        log.warning("GEMINI_API_KEY not set. LLM features disabled.")
        return None

    _MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        log.info(f"Gemini client ready (model: {_MODEL_NAME})")
        return _client
    except ImportError:
        log.error(
            "google-genai package not installed. "
            "Run: pip install google-genai"
        )
        return None
    except Exception as err:
        log.error(f"Failed to create Gemini client: {err}")
        return None


def reset():
    """Reset client state so next call re-initializes.
    Use after changing API key or model at runtime.
    """
    global _client, _init_attempted, _MODEL_NAME
    _client = None
    _init_attempted = False
    _MODEL_NAME = None


def test_connection():
    """Quick connectivity check. Returns a status message string."""
    client = _get_client()
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return "API key not configured — set GEMINI_API_KEY in .env"
        return "Failed to create Gemini client — check logs"

    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents="Reply with the single word OK",
            config={"temperature": 0, "max_output_tokens": 10},
        )
        if response and response.text:
            return "Gemini connected successfully"
        return "Gemini responded but returned empty text"
    except Exception as err:
        err_str = str(err)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            return "Gemini connected (rate-limited right now, will work shortly)"
        return f"Gemini connection error: {err_str[:200]}"


def is_available():
    """Return True if the Gemini client could be initialized."""
    return _get_client() is not None


def ask_gemini(prompt, temperature=0.3, max_tokens=1024):
    """Send a prompt to Gemini, return the text response.

    Includes rate-limiting and retry logic for the free tier.
    Returns None if Gemini is unavailable or all retries fail.
    """
    global _last_call_time

    client = _get_client()
    if client is None:
        return None

    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        # respect rate limit
        elapsed = time.time() - _last_call_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        try:
            response = client.models.generate_content(
                model=_MODEL_NAME,
                contents=prompt,
                config=config,
            )
            _last_call_time = time.time()

            if response and response.text:
                return response.text.strip()
            return None

        except Exception as err:
            _last_call_time = time.time()
            err_str = str(err)

            # rate-limited — wait and retry
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = min(30, 5 * attempt)
                log.warning(
                    f"Gemini rate-limited (attempt {attempt}/{_MAX_RETRIES}), "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            log.error(f"Gemini API error: {err_str[:300]}")
            return None

    log.error("Gemini: all retries exhausted (rate limited).")
    return None


def ask_gemini_json(prompt, temperature=0.2, max_tokens=512):
    """Send prompt, parse response as JSON. Returns dict/list or None."""
    full_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: Respond with valid JSON only. No markdown, no code fences, "
        "no explanation text before or after the JSON."
    )

    text = ask_gemini(full_prompt, temperature=temperature, max_tokens=max_tokens)
    if text is None:
        return None

    cleaned = text.strip()
    # strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning(f"Gemini returned non-JSON: {cleaned[:200]}")
        return None
