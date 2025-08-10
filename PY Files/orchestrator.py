# PY Files/orchestrator.py

from __future__ import annotations

import os
import re
import json
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Core deps (safe imports) ---
from llm_client import get_openai_client, chat_completion
from loader import load_master_metadata_index
from executor import run_intent_task
from phase1_ingest.pipeline import run_pipeline

import llm_prompts  # scaffold (llm_prompts.py)

# Try to use your external classifier if present; fall back to local one below
try:
    from intent import classify_intent as _external_classify_intent
except Exception:
    _external_classify_intent = None

# Tools runtime is loaded lazily inside _safe_tool_specs/_execute_plan
# to avoid import-time crashes.


# ----------------------------
# Light paths shim
# ----------------------------
class Paths:
    """
    Minimal shim to pass metadata folder & alias path.
    Update to your real project's Paths object if you have one.
    """
    def __init__(self, metadata_folder: str, alias_json: Optional[str] = None):
        self.metadata_folder = metadata_folder
        self.alias_json = alias_json  # optional explicit path


# ----------------------------
# Phase 1 wrapper
# ----------------------------
def run_ingest_pipeline(
    source: bytes | bytearray | str,
    filename: Optional[str],
    paths: Optional[Any],
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """
    Thin wrapper for Phase 1 ingestion. Returns (cleaned_sheets, metadata).
    """
    cleaned_sheets, metadata = run_pipeline(source=source, filename=filename, paths=paths)
    return cleaned_sheets, metadata


# ----------------------------
# Cloud tool-planned path
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
    intent_info = classify_user_intent(user_question)
    intent = intent_info["intent"]
    model_size = intent_info["model_size"]  # "small"|"large"
    model = "gpt-4o" if model_size == "large" else "gpt-4o-mini"

    # 1) Propose a tool plan (JSON) with GPT
    tools_catalog = _safe_tool_specs()
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
# Tool planning helpers
# ----------------------------
def _safe_tool_specs() -> Dict[str, Dict[str, Any]]:
    try:
        from tools_runtime import tool_specs  # lazy
        return tool_specs()
    except Exception as e:
        # Minimal fallback so the app still renders and we can see the error
        return {
            "_load_error": {
                "purpose": f"tools_runtime.tool_specs failed to import: {e}",
                "args_schema": {},
                "returns": "N/A",
            }
        }


def _propose_tool_plan(
    client,
    question: str,
    tools: Dict[str, Dict[str, Any]],
    cleansed_paths: List[str],
    app_paths: Any,
) -> List[Dict[str, Any]]:
    tool_guide = llm_prompts._render_tool_guide(tools)  # reuse private helper
    defaults = {
        "cleansed_paths": cleansed_paths,
        "artifact_folder_charts": getattr(
            app_paths, "dbx_charts_folder", getattr(app_paths, "dbx_eda_charts_folder", None)
        ),
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
    return [
        {
            "tool": "dataframe_query",
            "args": {
                "cleansed_paths": cleansed_paths[:1],
                "sheet": None,
                "filters": [],
                "groupby": None,
                "metrics": None,
                "limit": 50,
                "artifact_folder": getattr(app_paths, "dbx_summaries_folder", None),
            },
        }
    ]


def _execute_plan(plan: List[Dict[str, Any]], cleansed_paths: List[str], app_paths: Any) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {"last_rows": None}
    artifacts: List[str] = []

    try:
        from tools_runtime import dataframe_query, chart, kb_search  # lazy
    except Exception as e:
        calls.append({"tool": "import", "args": {}, "error": f"Failed to import tools_runtime: {e}"})
        return {"calls": calls, "context": context, "artifacts": artifacts}

    for step in plan:
        tool = (step.get("tool") or "").strip()
        args = step.get("args") or {}

        if tool == "dataframe_query":
            args.setdefault("cleansed_paths", cleansed_paths[:1])
            args.setdefault("limit", 50)
            args.setdefault("artifact_folder", getattr(app_paths, "dbx_summaries_folder", None))
            try:
                res = dataframe_query(**args)
            except Exception as e:
                res = {"error": f"dataframe_query failed: {e}"}
            context["last_rows"] = res.get("preview")
            if res.get("artifact_path"):
                artifacts.append(res["artifact_path"])
            calls.append(
                {"tool": tool, "args": args, "result_meta": {k: res.get(k) for k in ("rowcount", "artifact_path", "sheet_used", "error")}}
            )

        elif tool == "chart":
            rows = args.get("rows")
            if isinstance(rows, str) and rows.strip() == "$prev_rows":
                rows = context.get("last_rows") or []
            args["rows"] = rows
            args.setdefault(
                "artifact_folder",
                getattr(app_paths, "dbx_charts_folder", getattr(app_paths, "dbx_eda_charts_folder", None)),
            )
            args.setdefault("base_name", "chart")
            try:
                res = chart(**args)
            except Exception as e:
                res = {"error": f"chart failed: {e}"}
            if res.get("image_path"):
                artifacts.append(res["image_path"])
            calls.append({"tool": tool, "args": args, "result_meta": {"image_path": res.get("image_path"), "error": res.get("error")}})

        elif tool == "kb_search":
            try:
                res = kb_search(args.get("query", ""), int(args.get("k", 5)))
            except Exception as e:
                res = {"error": f"kb_search failed: {e}", "citations": []}
            calls.append({"tool": tool, "args": args, "result_meta": {"citations": res.get("citations", []), "error": res.get("error")}})
            context["kb_chunks"] = res.get("chunks", [])
            context["kb_citations"] = res.get("citations", [])

        else:
            calls.append({"tool": tool, "args": args, "error": "Unsupported tool"})

    return {"calls": calls, "context": context, "artifacts": artifacts}


def _summarize_exec(exec_result: Dict[str, Any]) -> Tuple[str, List[str]]:
    calls = exec_result["calls"]
    artifacts = exec_result["artifacts"]

    lines = []
    for c in calls:
        if c.get("error"):
            lines.append(f"Tool error: {c['error']}")
        elif c["tool"] == "dataframe_query":
            meta = c.get("result_meta", {})
            lines.append(f"Dataframe Query • rows={meta.get('rowcount','?')} • sheet={meta.get('sheet_used')}")

    if not lines:
        lines.append("No data operations executed.")

    return "\n".join(lines), artifacts


def _extract_kb(exec_result: Dict[str, Any]) -> Tuple[str, List[str]]:
    ctx = exec_result["context"]
    chunks = ctx.get("kb_chunks") or []
    cits = ctx.get("kb_citations") or []
    if not chunks:
        return "", []
    parts = []
    for ch in chunks[:2]:
        txt = ch.get("text") if isinstance(ch, dict) else str(ch)
        parts.append(txt.strip())
    return "\n\n---\n\n".join(parts), cits


# ----------------------------
# Intent classification (robust)
# ----------------------------
SUPPORTED_INTENTS = [
    "compare",
    "root_cause",
    "forecast",
    "summarize",
    "eda",
    "rank",
    "anomaly",
    "optimize",
    "filter",
]

WIP_EO_BIAS_KEYWORDS = {
    "wip": ["wip", "work in progress", "job", "shop order", "wo ", "work order"],
    "eo": ["e&o", "excess", "obsolete", "slow mover", "dead stock", "write-off", "write off"],
}

DEFAULT_INTENT_MODEL = os.getenv("INTENT_MODEL", "gpt-4o-mini")
CONFIDENCE_THRESHOLD = 0.52  # tune as needed


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


def _fallback_intent(user_question: str) -> Tuple[str, float]:
    q = user_question.lower()

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
    """
    Model-based classification with JSON parsing hardening + fallback.
    Also exposes 'model_size' for routing: 'small' vs 'large'.
    """
    # Prefer your external classifier if available
    if _external_classify_intent is not None:
        try:
            out = _external_classify_intent(user_question)
            # Normalize and add model_size hint if missing
            model_size = out.get("model_size") or ("large" if out.get("intent") in {"root_cause", "forecast"} else "small")
            out["model_size"] = model_size
            return out
        except Exception:
            pass

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
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
        data = json.loads(text)

        intent = str(data.get("intent", "")).lower().strip()
        confidence = float(data.get("confidence", 0.0))
        reasoning = data.get("reasoning", "")
        entities = data.get("entities", []) or []

        if intent not in SUPPORTED_INTENTS:
            intent, confidence = _fallback_intent(user_question)
            reasoning = reasoning or "Heuristic fallback triggered due to unsupported intent."

        model_size = "large" if intent in {"root_cause", "forecast"} else "small"

        return {
            "intent": intent,
            "confidence": max(0.0, min(confidence, 1.0)),
            "reasoning": reasoning,
            "entities": entities,
            "raw": data,
            "model_size": model_size,
        }
    except Exception as e:
        intent, confidence = _fallback_intent(user_question)
        model_size = "large" if intent in {"root_cause", "forecast"} else "small"
        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": f"Classifier exception fallback: {e}",
            "entities": [],
            "raw": None,
            "model_size": model_size,
        }


# ----------------------------
# Existing “Phase 4” entry (unchanged signature)
# ----------------------------
def run_query_pipeline(query: str, df_dict: dict, metadata_path: str) -> dict:
    """
    Orchestrates the full flow: classify → confidence gate → match/execute → return.
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
        abstained = {
            "intent": "abstain",
            "confidence": 0.0,
            "message": "I’m not confident enough to proceed without clarification.",
            "reason": f"Low confidence ({confidence:.2f}) for intent '{intent}'. "
                      f"Reason: {intent_result.get('reasoning','')}",
        }
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
    return {"data": result, "intent_classification": intent_result, "flags": flags}


# ----------------------------
# Tiny utils
# ----------------------------
def _is_wip_or_eo_query(user_question: str) -> bool:
    q = user_question.lower()
    return any(k in q for k in WIP_EO_BIAS_KEYWORDS["wip"] + WIP_EO_BIAS_KEYWORDS["eo"])

