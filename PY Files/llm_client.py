# PY Files/llm_client.py

import os
from typing import Optional

# Optional import if running inside Streamlit
try:
    import streamlit as st  # type: ignore
    _SECRETS = dict(getattr(st, "secrets", {}))
except Exception:
    _SECRETS = {}

def _get_api_key() -> str:
    """
    Prefer Streamlit secrets, then env var.
    """
    key = _SECRETS.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not configured. Add it to Streamlit secrets "
            "(.streamlit/secrets.toml) or set it as an environment variable."
        )
    return key

def get_openai_client():
    """
    Return an OpenAI v1 client, or raise a clear error.
    """
    api_key = _get_api_key()
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

def chat_completion(client, messages, model: str = "gpt-4o-mini") -> str:
    """
    Compatibility wrapper:
    - Try Chat Completions
    - Fall back to Responses API
    Returns assistant text.
    """
    # Try Chat Completions first
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # Fallback: Responses API
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": messages[-1]["content"]}],
                temperature=0.2,
            )
            # Prefer the convenience property if present
            text = getattr(resp, "output_text", None)
            if text:
                return text.strip()
            # Otherwise scan structured output
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "output_text":
                    return (getattr(item, "text", "") or "").strip()
            return ""
        except Exception as e2:
            return f"⚠️ GPT summary failed: {e2}"

