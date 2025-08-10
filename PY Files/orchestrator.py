# orchestrator.py

import os
import re
import json
from typing import Optional, Any, Tuple, Dict
import pandas as pd

from llm_client import get_openai_client
from loader import load_master_metadata_index
from executor import run_intent_task
from phase1_ingest.pipeline import run_pipeline

# PY Files/orchestrator.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import traceback

from llm_client import get_openai_client, chat_completion
import llm_prompts  # your scaffold (renamed from llm/prompts.py to llm_prompts.py)
from intent import classify_intent
from tools_runtime import tool_specs, dataframe_query, chart, kb_search

class Paths:
    """
    Minimal shim to pass metadata folder & alias path.
    Update to your real project's Paths object if you have one.
    """
    def __init__(self, metadata_folder: str, alias_json: Optional[str] = None):
        self.metadata_folder = metadata_folder
        self.alias_json = alias_json  # optional explicit path

def run_ingest_pipeline(
    source: bytes | bytearray | str,
    filename: Optional[str],
    paths: Optional[Any]
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """
    Thin wrapper for Phase 1 ingestion. Returns (cleaned_sheets, metadata).
    """
    cleaned_sheets, metadata = run_pipeline(source=source, filename=filename, paths=paths)
    return cleaned_sheets, metadata

# ----------------------------
# Public API
# ----------------------------

def answer_question(
    user_question: str,
    *,
    app_paths: Any,
    cleansed_paths: List[str],
    answer_style: str = llm_prompts.ANSWER_STYLE_CONCISE,
) -> Dict[str, Any]:
    """
    Orchestrate: classify intent -> propose tool plan -> execute -> compose final answer.
    Cloud-only; cleansed_paths are Dropbox paths to .xlsx produced by Phase 1.
    Returns dict with final_text, intent_info, tool_calls, artifacts, debug.
    """
    # 0) Intent + model routing
    intent_info = classify_intent(user_question)
    intent = intent_info["intent"]
    model_size = intent_info["model_size"]  # "small"|"large"
    model = "gpt-4o" if model_size == "large" else "gpt-4o-mini"

    # 1) Propose a tool plan (JSON) with GPT
    tools_catalog = tool_specs()
    client = get_openai_client()
    plan = _propose_tool_plan(client, user_question, tools_catalog, cleansed_paths, app_paths)

    # 2) Execute plan (guarded)
    exec_result = _execute_plan(plan, cleansed_paths, app_paths)

    # Build quantitative context (compact text for scaffold)
    quantitative_context, matched_artifacts = _summarize_exec(exec_result)

    # 3) (Optional) KB enrich if plan included kb_search or we want enrichment
    kb_ctx, kb_citations = _extract_kb(exec_result)

    # 4) Compose final answer with scaffold
    messages = llm_prompts.build_scaffold_messages(
        user_question=user_question,
        intent=intent,
        tools_catalog=tools_catalog,
        quantitative_context=quantitative_context,
        kb_context=kb_ctx,
        matched_artifacts=matched_artifacts,
        matched_docs=kb_citations,
        answer_style=answer_style,
        model_size=model_size,
    )
    final_text = chat_completion(client, messages, model=model)

    return {
        "final_text": final_text,
        "intent_info": intent_info,
        "tool_calls": exec_result["calls"],
        "artifacts": matched_artifacts,
        "kb_citations": kb_citations,
        "debug": {
            "plan_raw": plan,
            "quantitative_context": quantitative_context,
            "kb_context": kb_ctx,
        },
    }

# ----------------------------
# Internal helpers
# ----------------------------

def _propose_tool_plan(client, question: str, tools: Dict[str, Dict[str, Any]], cleansed_paths: List[str], app_paths: Any) -> List[Dict[str, Any]]:
    """
    Ask GPT for a minimal JSON plan of tool calls. We provide guardrails and defaults.
    """
    tool_guide = llm_prompts._render_tool_guide(tools)  # private helper is fine to reuse
    defaults = {
        "cleansed_paths": cleansed_paths,
        "artifact_folder_charts": getattr(app_paths, "dbx_charts_folder", getattr(app_paths, "dbx_eda_charts_folder", None)),
        "artifact_folder_summaries": getattr(app_paths, "dbx_summaries_folder", None),
    }
    prompt = (
        "Plan a minimal set of tool calls to answer the user's question.\n"
        "Return STRICT JSON (list of steps). Allowed tools and args schema are below.\n"
        "Rules:\n"
        "- Use only these tools: dataframe_query, chart, kb_search.\n"
        "- Prefer a single dataframe_query; only add chart if a visual would help.\n"
        "- Always include 'artifact_folder' when saving outputs.\n"
        "- Use the provided 'cleansed_paths' for data.\n"
        "- If the question asks for guidance/policy/SOP, include a kb_search step.\n"
        "- Keep args small; do not include huge literal data.\n\n"
        f"{tool_guide}\n\n"
        f"Defaults (fill missing args with these):\n{json.dumps(defaults, ensure_ascii=False)}\n\n"
        f"USER QUESTION: {question}\n\n"
        "Respond with JSON only. Example:\n"
        '[{"tool":"dataframe_query","args":{"cleansed_paths":["dropbox://...xlsx"],"sheet":"Aged WIP","filters":[],"groupby":["plant"],"metrics":[{"col":"extended_cost","agg":"sum"}],"limit":25,"artifact_folder":"dropbox://.../03_Summaries"}},'
        '{"tool":"chart","args":{"kind":"bar","rows":"$prev_rows","x":"plant","y":["extended_cost_sum"],"title":"Cost by Plant","artifact_folder":"dropbox://.../02_EDA_Charts","base_name":"cost_by_plant"}}]'
    )
    messages = [
        {"role": "system", "content": "You are a planning assistant that outputs valid JSON for tool execution, nothing else."},
        {"role": "user", "content": prompt},
    ]
    plan_text = chat_completion(client, messages, model="gpt-4o-mini")
    try:
        plan = json.loads(plan_text)
        if isinstance(plan, list):
            return plan
    except Exception:
        pass
    # Fallback trivial plan
    return [{
        "tool": "dataframe_query",
        "args": {
            "cleansed_paths": cleansed_paths[:1],
            "sheet": None,
            "filters": [],
            "groupby": None,
            "metrics": None,
            "limit": 50,
            "artifact_folder": getattr(app_paths, "dbx_summaries_folder", None),
        }
    }]

def _execute_plan(plan: List[Dict[str, Any]], cleansed_paths: List[str], app_paths: Any) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {"last_rows": None}
    artifacts: List[str] = []

    for step in plan:
        tool = (step.get("tool") or "").strip()
        args = step.get("args") or {}

        # supply defaults
        if tool == "dataframe_query":
            args.setdefault("cleansed_paths", cleansed_paths[:1])
            args.setdefault("limit", 50)
            args.setdefault("artifact_folder", getattr(app_paths, "dbx_summaries_folder", None))
            res = dataframe_query(**args)
            # keep a small preview for potential chart
            context["last_rows"] = res.get("preview")
            if res.get("artifact_path"):
                artifacts.append(res["artifact_path"])
            calls.append({"tool": tool, "args": args, "result_meta": {k: res.get(k) for k in ("rowcount","artifact_path","sheet_used")}})

        elif tool == "chart":
            # allow "$prev_rows" to reference last preview rows
            rows = args.get("rows")
            if isinstance(rows, str) and rows.strip() == "$prev_rows":
                rows = context.get("last_rows") or []
            args["rows"] = rows
            args.setdefault("artifact_folder", getattr(app_paths, "dbx_charts_folder", getattr(app_paths, "dbx_eda_charts_folder", None)))
            args.setdefault("base_name", "chart")
            res = chart(**args)
            if res.get("image_path"):
                artifacts.append(res["image_path"])
            calls.append({"tool": tool, "args": args, "result_meta": {"image_path": res.get("image_path")}})

        elif tool == "kb_search":
            res = kb_search(args.get("query",""), int(args.get("k", 5)))
            calls.append({"tool": tool, "args": args, "result_meta": {"citations": res.get("citations", [])}})
            # store for enrichment
            context["kb_chunks"] = res.get("chunks", [])
            context["kb_citations"] = res.get("citations", [])

        else:
            calls.append({"tool": tool, "args": args, "error": "Unsupported tool"})

    return {"calls": calls, "context": context, "artifacts": artifacts}

def _summarize_exec(exec_result: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Build a compact quantitative context string + artifact paths for the scaffold.
    """
    calls = exec_result["calls"]
    ctx = exec_result["context"]
    artifacts = exec_result["artifacts"]

    # Try to render the last dataframe preview as a tiny table in text
    lines = []
    for c in calls:
        if c["tool"] == "dataframe_query":
            meta = c.get("result_meta", {})
            lines.append(f"Dataframe Query • rows={meta.get('rowcount','?')} • sheet={meta.get('sheet_used')}")
            # keep it compact; show first 3 rows if available
            # (Rows are already safe JSON dicts)
            # Note: defer full tables to the saved artifact
    if not lines:
        lines.append("No data operations executed.")

    return "\n".join(lines), artifacts

def _extract_kb(exec_result: Dict[str, Any]) -> Tuple[str, List[str]]:
    ctx = exec_result["context"]
    chunks = ctx.get("kb_chunks") or []
    cits = ctx.get("kb_citations") or []
    if not chunks:
        return "", []
    # Compact the first couple of chunks
    parts = []
    for ch in chunks[:2]:
        txt = ch.get("text") if isinstance(ch, dict) else str(ch)
        parts.append(txt.strip())
    return "\n\n---\n\n".join(parts), cits

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

