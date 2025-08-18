# PY Files/phase4_knowledge/response_composer.py
# Phase 4.3 — Response Composer
# Takes a query + packed context, sends to OpenAI chat model, returns answer.

from __future__ import annotations
import os
from typing import Optional
from openai import OpenAI

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

def _load_openai_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in Streamlit secrets or environment.")
    return key

def _get_client() -> OpenAI:
    return OpenAI(api_key=_load_openai_key())

def compose_response(
    query: str,
    context: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """
    Compose a model response given the user query and packed KB context.
    """
    client = _get_client()
    sys_msg = system_prompt or (
        "You are a helpful assistant answering the user using the provided context. "
        "Only use the information in the context to answer. If the context is insufficient, say so."
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# CLI for quick testing
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Response Composer")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = ap.parse_args()

    answer = compose_response(args.query, args.context, model=args.model)
    print("\n=== ANSWER ===\n")
    print(answer)
