# orchestrator.py

import os
import re
import json
from typing import Dict, Any, Tuple

from llm_client import get_openai_client
from loader import load_master_metadata_index
from executor import run_intent_task

# --- Supported intents & lightweight biasing --------------------------------
SUPPORTED_INTENTS = [
    "compare", "root_cause", "forecast", "summarize",
    "eda", "rank", "anomaly", "optimize", "filter"
]

WIP_EO_BIAS_KEYWORDS = {
    "wip": ["wip", "work in progress", "job", "shop order", "wo ", "work order"],
    "eo": ["e&o", "excess", "obsolete", "slow mover", "dead stock", "write-off", "write off"]
}

DEFAULT_INTENT_MODEL = os.getenv("INTENT_MODEL", "gpt-4o-mini")
CONFIDENCE_THRESHOLD = 0.52  # tune as needed


# --- Prompt builder ----------------------------------------------------------
def _build_intent_prompt(user_question: str) -> str:
    supported = ", ".join(SUPPORTED_INTENTS)
    return f"""
You are an intent classifier for a supply-chain analytics assistant.

Return STRICT JSON with keys: intent, confidence (0-1), reasoning, entities (list).
Choose one intent from: [{supported}].

Rules:
- Prefer "root_cause" for WIP/E&O investigations.
- "compare" when asking differences/changes across periods/files.
- "forecast" for demand/ROP/par/safety stock/seasonality/future E&O.
- "eda" for general profiling, distributions, outliers.
- "rank" for prioritization/scoring (e.g., top risky SKUs).
- "anomaly" for spikes/drops/unexpected values.
- "optimize" for policy/parameter/procedure recommendations.
- "filter" for constrained views (country, plant, family).

Question: {user_question}
Return JSON only.
""".strip()


# --- Classifier with robust fallback ----------------------------------------
def _fallback_intent(user_question: str) -> Tuple[str, float]:
    q = user_question.lower()

    # WIP/E&O bias for root cause
    if any(k in q for k in WIP_EO_BIAS_KEYWORDS["wip"]) or any(k in q for k in WIP_EO_BIAS_KEYWORDS["eo"]):
        if any(w in q for w in ["why", "cause", "driver", "increase", "rise", "spike", "root"]):
            return "root_cause", 0.62

    if any(w in q for w in ["compare", "difference", "delta", "vs", "change", "trend vs"]):
        return "compare", 0.58
    if any(w in q for w in ["forecast", "predict", "par", "reorder point", "rop", "safety stock", "seasonal"]):
        return "forecast", 0.58
    if any(w in q for w in ["profile", "distribution", "eda", "summary stats", "outlier"]):
        return "eda", 0.55
    if any(w in q for w in ["rank", "top", "prioritize", "score", "highest"]):
        return "rank", 0.55
    if any(w in q for w in ["anomaly", "unexpected", "spike", "drop", "weird"]):
        return "anomaly", 0.55
    if any(w in q for w in ["optimize", "policy", "procedure", "parameter", "improve"]):
        return "optimize", 0.55
    if any(w in q for w in ["filter", "only show", "just", "subset", "limit to"]):
        return "filter", 0.55
    if any(w in q for w in ["summarize", "tl;dr", "brief", "overview"]):
        return "summarize", 0.55

    return "eda", 0.45


def classify_user_intent(user_question: str, client=None) -> Dict[str, Any]:
    """Model-based classification with JSON parsing hardening + fallback."""
    if client is None:
        client = get_openai_client()
    prompt = _build_intent_prompt(user_question)

    try:
        resp = client.responses.create(
            model=DEFAULT_INTENT_MODEL,
            input=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.output_text.strip()
        # Strip accidental code fences
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
        data = json.loads(text)

        intent = str(data.get("intent", "")).lower().strip()
        confidence = float(data.get("confidence", 0.0))
        reasoning = data.get("reasoning", "")
        entities = data.get("entities", []) or []

        if intent not in SUPPORTED_INTENTS:
            intent, confidence = _fallback_intent(user_question)
            reasoning = reasoning or "Heuristic fallback triggered due to unsupported intent."

        return {
            "intent": intent,
            "confidence": max(0.0, min(confidence, 1.0)),
            "reasoning": reasoning,
            "entities": entities,
            "raw": data,
        }
    except Exception as e:
        # Model failure → fallback
        intent, confidence = _fallback_intent(user_question)
        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": f"Classifier exception fallback: {e}",
            "entities": [],
            "raw": None,
        }


# --- Utility -----------------------------------------------------------------
def _is_wip_or_eo_query(user_question: str) -> bool:
    q = user_question.lower()
    return any(k in q for k in WIP_EO_BIAS_KEYWORDS["wip"] + WIP_EO_BIAS_KEYWORDS["eo"])


def _abstain(reason: str) -> Dict[str, Any]:
    return {
        "intent": "abstain",
        "confidence": 0.0,
        "message": "I’m not confident enough to proceed without clarification.",
        "reason": reason,
    }


# --- Public entry point (keeps your original signature) ----------------------
def run_query_pipeline(query: str, df_dict: dict, metadata_path: str) -> dict:
    """
    Orchestrates the full flow: classify → confidence gate → match/execute → return.

    Args:
        query (str): User's natural language question.
        df_dict (dict): Dict of DataFrames keyed by (filename, sheetname).
        metadata_path (str): Path to master_metadata_index.json.

    Returns:
        dict: Structured result (original executor result + intent metadata).
    """
    # 1) Clients & metadata
    client = get_openai_client()
    metadata_index = load_master_metadata_index(metadata_path)

    # 2) Intent classification (no dropdowns)
    intent_result = classify_user_intent(query, client=client)
    intent = intent_result["intent"]
    confidence = intent_result["confidence"]

    # 3) Confidence gate (abstain if too low)
    if confidence < CONFIDENCE_THRESHOLD:
        abstained = _abstain(f"Low confidence ({confidence:.2f}) for intent '{intent}'. Reason: {intent_result['reasoning']}")
        # include classification context for visibility
        abstained["intent_classification"] = intent_result
        abstained["flags"] = {"wip_eo_bias": _is_wip_or_eo_query(query)}
        return abstained

    # 4) Optional flags for downstream (executor can ignore if not used)
    flags = {"wip_eo_bias": _is_wip_or_eo_query(query)}

    # 5) Execute your existing pipeline (no signature change)
    result = run_intent_task(query, df_dict, metadata_index, client)

    # 6) Attach classification + flags for transparency (non-breaking)
    if isinstance(result, dict):
        result.setdefault("intent_classification", intent_result)
        result.setdefault("flags", flags)
        return result

    # Fallback shape if executor returns unexpected type
    return {
        "data": result,
        "intent_classification": intent_result,
        "flags": flags,
    }

