# PY Files/router_client.py
from __future__ import annotations
import os, json
from typing import Dict, Any
from openai import OpenAI

# Minimal, self-contained Router caller via Chat Completions.
# We embed the same rules you pasted into the GPT Builder so this works without a GPT ID.

ROUTER_SYSTEM_PROMPT = """
You are the Router for SCIE Ethos.
Your only job is to classify the user's request and return a routing CONTRACT as strict JSON.
Never write prose, markdown, or code fences. Output JSON only.

ALLOWED INTENTS (top-level):
- "doc_reader"     → meetings/docs/transcripts → insights + evidence hints
- "forecast"       → demand projection, seasonality, safety stock, ROP, par
- "rca"            → root cause, comparison, scenario, optimization, anomalies

ACTIONS (select the MINIMAL set needed):
- "kb_search"         (vector/RAG over docs)
- "metadata_select"   (file resolve, versioning, shorthand like R401)
- "forecast_demand"   (when forecasting/policy math is required)

ROUTING RULES
- Choose the single best intent. If the query mixes tasks, pick the dominant one.
- If information is missing, list crisp follow-ups in "missing".
- Fill "files_hint" with 0–3 keywords that help locate files.
- Keep "actions" short (usually 1–2).
- Set "confidence" 0.00–1.00 based on clarity + match to allowed intents.
- Never invent entities, file names, dates, or numbers.
- Do not execute work—just route.

CONTRACT (return EXACTLY these keys; types must match):
{
  "intent": "doc_reader | forecast | rca",
  "files_hint": ["optional string", "optional string"],
  "params": {
    "entity": "optional string (e.g., part/plant/customer)",
    "period": "optional string (e.g., Q2-2025 or 2024-01..2024-12)",
    "granularity": "optional string: job | part | month | plant"
  },
  "actions": ["kb_search", "metadata_select"],
  "limits": { "latency_s": 8, "cost_usd_max": 0.10 },
  "missing": ["optional follow-up question"],
  "why": "1–2 sentence router rationale.",
  "confidence": 0.85
}

OUTPUT POLICY
- Respond with a SINGLE JSON object only.
- No trailing comments, no backticks, no extra keys, no nulls (omit absent fields).
"""

class RouterClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def classify(self, user_text: str) -> Dict[str, Any]:
        if not os.environ.get("OPENAI_API_KEY"):
            # Safe fallback for dev without keys
            return {
                "intent": "rca",
                "files_hint": [],
                "params": {},
                "actions": ["metadata_select","kb_search"],
                "limits": {"latency_s": 8, "cost_usd_max": 0.10},
                "missing": [],
                "why": "Fallback route (no API key set).",
                "confidence": 0.4
            }

        # Include learned patterns if memory is enabled
        memory_block = ""
        if os.getenv("MEMORY_ENABLED", "0") in ("1","true","True"):
            try:
                from memory_store import get_router_context
                memory_block = get_router_context(max_items=40)
            except Exception:
                pass  # memory failure should not break routing

        system_prompt = ROUTER_SYSTEM_PROMPT
        if memory_block:
            system_prompt += "\n\n" + memory_block

        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_text}
            ],
        )
        try:
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            raise ValueError(f"Router returned invalid JSON: {e}")