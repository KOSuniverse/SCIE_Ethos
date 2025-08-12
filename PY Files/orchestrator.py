# PY Files/orchestrator.py
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Core deps (safe imports, with fallbacks) ---
try:
    from llm_client import get_openai_client, chat_completion
except Exception:
    # Minimal fallbacks to avoid import errors during boot
    def get_openai_client():
        raise RuntimeError("llm_client.get_openai_client not available")

    def chat_completion(client, messages, model: str):
        raise RuntimeError("llm_client.chat_completion not available")

# Optional: file listing utility (auto-scan cleansed files)
try:
    from file_utils import list_cleaned_files  # your helper in main.py context
except Exception:
    def list_cleaned_files() -> List[str]:
        # Fallback: look at a conventional folder if mounted or pre-synced
        # If your runtime doesn't support local listing, pass cleansed_paths from UI
        return []

# Phase 1 ingest passthrough
try:
    from phase1_ingest.pipeline import run_pipeline
except Exception:
    def run_pipeline(*args, **kwargs):
        raise RuntimeError("phase1_ingest.pipeline.run_pipeline not available")

# Phase 4 executor (your business logic once an intent is chosen)
try:
    from executor import run_intent_task
except Exception:
    def run_intent_task(query, df_dict, metadata_index, client):
        return {"error": "executor.run_intent_task unavailable"}

# Metadata loader (master index)
try:
    from loader import load_master_metadata_index
except Exception:
    def load_master_metadata_index(path: str):
        return {}

# Prompt scaffolds / system prompt text
import llm_prompts  # your existing scaffold (must exist)

# --- Optional external classifier (if present) ---
try:
    from intent import classify_intent as _external_classify_intent
except Exception:
    _external_classify_intent = None


# =============================================================================
# Public: thin wrapper for Phase 1 ingest (unchanged)
# =============================================================================
def run_ingest_pipeline(
    source: bytes | bytearray | str,
    filename: Optional[str],
    paths: Optional[Any],
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    cleaned_sheets, metadata = run_pipeline(source=source, filename=filename, paths=paths)
    return cleaned_sheets, metadata


# =============================================================================
# Artifact location routing (Dropbox vs S3)
# =============================================================================
def _artifact_backend() -> str:
    return (os.getenv("ARTIFACT_BACKEND") or "dropbox").strip().lower()

def _s3_bucket() -> Optional[str]:
    return os.getenv("S3_BUCKET")

def _s3_prefix() -> str:
    return (os.getenv("S3_PREFIX") or "project-root").strip().strip("/")

def _dbx_default_charts(app_paths: Any) -> Optional[str]:
    return getattr(app_paths, "dbx_charts_folder", 
                   getattr(app_paths, "dbx_eda_charts_folder", 
                          getattr(app_paths, "default_charts_path", "/04_Data/02_EDA_Charts")))

def _dbx_default_summaries(app_paths: Any) -> Optional[str]:
    return getattr(app_paths, "dbx_summaries_folder", 
                   getattr(app_paths, "default_summaries_path", "/04_Data/03_Summaries"))

def _artifact_folder(kind: str, app_paths: Any) -> Optional[str]:
    """
    kind: 'charts' | 'summaries'
    Returns a folder URI/path appropriate for the configured backend.
    """
    backend = _artifact_backend()
    if backend == "s3":
        bucket = _s3_bucket()
        if not bucket:
            raise RuntimeError("ARTIFACT_BACKEND=s3 requires S3_BUCKET env")
        base = f"s3://{bucket}/{_s3_prefix()}"
        if kind == "charts":
            charts_path = getattr(app_paths, "s3_charts_path", "04_Data/02_EDA_Charts")
            return f"{base}/{charts_path}"
        summaries_path = getattr(app_paths, "s3_summaries_path", "04_Data/03_Summaries")
        return f"{base}/{summaries_path}"
    # default dropbox pathing
    return _normalize_dbx_path(_dbx_default_charts(app_paths) if kind == "charts" else _dbx_default_summaries(app_paths))


# =============================================================================
# Dropbox/S3 path normalization
# =============================================================================
def _normalize_dbx_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    s = p.strip()
    if s.startswith(("/", "id:", "rev:", "ns:")):
        return s
    if s.startswith("dropbox://") or s.startswith("dbx://"):
        s = s.split("://", 1)[1]
    if not s.startswith("/"):
        s = "/" + s
    return s

def _normalize_dbx_paths(paths: Optional[List[str]]) -> List[str]:
    return [_normalize_dbx_path(p) for p in (paths or []) if _normalize_dbx_path(p)]


# =============================================================================
# COMBINED CONTEXT Orchestration (Excel + KB, no dropdowns)
# =============================================================================
def answer_question(
    user_question: str,
    *,
    app_paths: Any,
    cleansed_paths: Optional[List[str]] = None,
    answer_style: str = llm_prompts.ANSWER_STYLE_CONCISE,
) -> Dict[str, Any]:
    """
    Enterprise path:
      1) Auto-intent
      2) Build global context (ALL cleansed files + sheets + columns; KB candidates)
      3) Plan with GPT (NO guessing; must pick from provided lists)
      4) Execute with guardrails (sheet/file validation, alias-aware args)
      5) Compose final answer with both quantitative (Excel) + qualitative (KB) context
    """
    # 0) Intent + model routing
    intent_info = classify_user_intent(user_question)
    intent = intent_info["intent"]
    model_size = intent_info.get("model_size", "large" if intent in {"root_cause", "forecast"} else "small")
    model = "gpt-4o" if model_size == "large" else "gpt-4o-mini"
    
    print(f"🧠 ORCHESTRATOR DEBUG: Intent={intent}, Model={model}, Confidence={intent_info.get('confidence')}")

    client = get_openai_client()

    # 1) Assemble global context
    excel_ctx = _build_excel_context(cleansed_paths)   # files -> sheets (+types) (+columns)
    kb_ctx_preview = _kb_candidates(user_question)     # doc titles or brief refs
    
    print(f"📊 ORCHESTRATOR DEBUG: Excel files found: {len(excel_ctx.get('files', []))}")
    print(f"📚 ORCHESTRATOR DEBUG: KB candidates: {len(kb_ctx_preview)}")

    # 2) Propose a plan with hard guardrails
    tools_catalog = _safe_tool_specs()
    print(f"🔧 ORCHESTRATOR DEBUG: Available tools: {list(tools_catalog.keys())}")
    
    plan = _propose_tool_plan(
        client=client,
        question=user_question,
        tools=tools_catalog,
        excel_context=excel_ctx,
        kb_candidates=kb_ctx_preview,
        app_paths=app_paths,
    )
    
    print(f"📋 ORCHESTRATOR DEBUG: Proposed plan has {len(plan)} steps")
    for i, step in enumerate(plan):
        print(f"   Step {i+1}: {step.get('tool', 'unknown')} - {step.get('purpose', 'no purpose')}")

    # 3) Execute plan (guarded)
    exec_result = _execute_plan(plan, app_paths)
    print(f"⚡ ORCHESTRATOR DEBUG: Execution completed with {len(exec_result)} results")

    # Build quantitative context (compact)
    quantitative_context, matched_artifacts = _summarize_exec(exec_result)
    # Extract KB chunks for final answer
    kb_text_ctx, kb_citations = _extract_kb(exec_result)

    # 4) Compose final answer
    messages = llm_prompts.build_scaffold_messages(
        user_question=user_question,
        intent=intent,
        tools_catalog=tools_catalog,
        quantitative_context=quantitative_context,
        kb_context=kb_text_ctx,
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
            "orchestrator_flow": {
                "intent_classification": intent_info,
                "model_selected": model,
                "tools_available": list(tools_catalog.keys()),
                "plan_steps": len(plan),
                "execution_results": len(exec_result["calls"]),
                "excel_files_found": len(excel_ctx.get("files", [])),
                "kb_candidates_found": len(kb_ctx_preview),
            },
            "plan_raw": plan,
            "excel_context": excel_ctx,
            "kb_candidates": kb_ctx_preview,
            "quantitative_context": quantitative_context,
            "execution_details": exec_result,
        },
    }


# =============================================================================
# Context builders
# =============================================================================
def _build_excel_context(cleansed_paths: Optional[List[str]]) -> Dict[str, Any]:
    """
    Returns a dict:
    {
      "files": ["dropbox://cleansed_folder/a.xlsx", ...],
      "sheets_by_file": {file: ["Inventory", "Aged WIP", ...]},
      "sheet_types": {file: {"Inventory":"inventory", "Aged WIP":"wip"}},
      "columns_by_sheet": {file: {"Aged WIP": ["job","extended_cost",...], ...}}  # if available
    }
    """
    # 1) pick files: use provided OR auto-scan
    files = _normalize_dbx_paths(cleansed_paths) if cleansed_paths else _normalize_dbx_paths(list_cleaned_files())

    sheets_by_file: Dict[str, List[str]] = {}
    sheet_types: Dict[str, Dict[str, str]] = {}
    columns_by_sheet: Dict[str, Dict[str, List[str]]] = {}

    # Lazy import tools_runtime helpers if available
    try:
        from tools_runtime import list_sheets, peek_columns  # optional helpers
    except Exception:
        list_sheets = None
        peek_columns = None

    for f in files:
        sheets = []
        try:
            if callable(list_sheets):
                sheets = list_sheets(f) or []
        except Exception:
            sheets = []
        sheets_by_file[f] = sheets

        # lightweight type hints by heuristics
        tmap: Dict[str, str] = {}
        for s in sheets:
            sl = s.lower()
            if any(k in sl for k in ("wip", "aged wip", "g512", "work in progress")):
                tmap[s] = "wip"
            elif any(k in sl for k in ("inventory", "finished goods", "fg")):
                tmap[s] = "inventory"
            else:
                tmap[s] = "unknown"
        sheet_types[f] = tmap

        # columns (optional; speeds planner correctness)
        cmap: Dict[str, List[str]] = {}
        if callable(peek_columns):
            for s in sheets:
                try:
                    cmap[s] = peek_columns(f, s) or []
                except Exception:
                    pass
        columns_by_sheet[f] = cmap

    return {
        "files": files,
        "sheets_by_file": sheets_by_file,
        "sheet_types": sheet_types,
        "columns_by_sheet": columns_by_sheet,
    }


def _kb_candidates(question: str, k: int = 6) -> List[str]:
    """
    Returns top-N KB titles/ids to nudge planner. Execution will still call kb_search.
    This abstracts FAISS vs future backends (OpenAI File Search, pgvector, etc.).
    """
    try:
        from tools_runtime import kb_suggest_titles  # optional preview function
        titles = kb_suggest_titles(question, k)
        return titles or []
    except Exception:
        # If no preview API, just return empty; executor can still run kb_search
        return []


# =============================================================================
# Planning (hard guardrails, no guessing)
# =============================================================================
def _safe_tool_specs() -> Dict[str, Dict[str, Any]]:
    try:
        from tools_runtime import tool_specs
        return tool_specs()
    except Exception as e:
        return {"_load_error": {"purpose": f"tools_runtime.tool_specs failed: {e}", "args_schema": {}, "returns": "N/A"}}

def _propose_tool_plan(
    client,
    question: str,
    tools: Dict[str, Dict[str, Any]],
    excel_context: Dict[str, Any],
    kb_candidates: List[str],
    app_paths: Any,
) -> List[Dict[str, Any]]:
    tool_guide = llm_prompts._render_tool_guide(tools)

    # Build artifact defaults based on backend
    defaults = {
        "artifact_folder_charts": _artifact_folder("charts", app_paths),
        "artifact_folder_summaries": _artifact_folder("summaries", app_paths),
        "excel": {
            "files": excel_context.get("files", []),
            "sheets_by_file": excel_context.get("sheets_by_file", {}),
            "sheet_types": excel_context.get("sheet_types", {}),
            "columns_by_sheet": excel_context.get("columns_by_sheet", {}),
        },
        "kb_candidates": kb_candidates,
        "rules": {
            "choose_files_from_list_only": True,
            "choose_sheets_from_list_only": True,
            "omit_sheet_if_unsure": True,
            "include_kb_if_candidates_present": True,
            "if_inventory_and_wip": "query both, summarize together",
        },
    }

    prompt = (
        "You are a planning assistant that emits STRICT JSON (a list of steps). "
        "Use only the tools and schema below. DO NOT guess file or sheet names — "
        "pick only from the provided context.\n\n"
        f"{tool_guide}\n\n"
        "Context:\n"
        f"{json.dumps(defaults, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "- Use: dataframe_query, chart, kb_search ONLY.\n"
        "- If both Inventory and WIP are relevant, include two dataframe_query steps and then a chart or combined summary.\n"
        "- Only use files from context.excel.files.\n"
        "- If you specify 'sheet', it MUST be in context.excel.sheets_by_file[file]. If unsure, omit 'sheet'.\n"
        "- If kb_candidates not empty and the question is analytical/interpretive, include a kb_search step.\n"
        "- Keep args small; do not include large literal data.\n\n"
        f"USER QUESTION: {question}\n\n"
        "Respond with JSON ONLY."
    )

    messages = [
        {"role": "system", "content": "Return JSON only. No prose."},
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
    files = excel_context.get("files", [])
    return [
        {
            "tool": "dataframe_query",
            "args": {
                "files": files[:1],
                "sheet": None,
                "filters": [],
                "groupby": None,
                "metrics": None,
                "limit": 50,
                "artifact_folder": _artifact_folder("summaries", app_paths),
            },
        }
    ]


# =============================================================================
# Execution with guardrails (validates file/sheet, normalizes paths, merges KB)
# =============================================================================
def _execute_plan(plan: List[Dict[str, Any]], app_paths: Any) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {"last_rows": None}
    artifacts: List[str] = []

    try:
        from tools_runtime import dataframe_query, chart, kb_search, list_sheets
    except Exception as e:
        calls.append({"tool": "import", "args": {}, "error": f"Failed to import tools_runtime: {e}"})
        return {"calls": calls, "context": context, "artifacts": artifacts}

    for step in plan:
        tool = (step.get("tool") or "").strip()
        args = step.get("args") or {}

        if tool == "dataframe_query":
            # Support new schema: {"files":[...]} or legacy {"cleansed_paths":[...]}
            files = args.get("files") or args.get("cleansed_paths") or []
            files = _normalize_dbx_paths(files)
            if not files:
                calls.append({"tool": tool, "args": args, "error": "No files provided"})
                continue

            # Validate requested sheet
            requested = (args.get("sheet") or "").strip() or None
            sheet_used = None
            try:
                available = list_sheets(files[0]) or []
            except Exception:
                available = []

            if requested and requested not in available:
                # Deterministic fallback: prefer WIP -> Inventory -> first
                sheet_used = _pick_fallback_sheet(available, requested)
            else:
                sheet_used = requested or _pick_fallback_sheet(available, None)

            # Artifact folder by backend
            artifact_folder = args.get("artifact_folder")
            if not artifact_folder:
                artifact_folder = _artifact_folder("summaries", app_paths)

            # Safe defaults
            args_exec = {
                "cleansed_paths": files[:1],  # current runtime supports single-file ops; planner can add more steps
                "sheet": sheet_used,
                "filters": args.get("filters", []),
                "groupby": args.get("groupby"),
                "metrics": args.get("metrics"),
                "limit": int(args.get("limit", 50)),
                "artifact_folder": artifact_folder,
            }

            try:
                res = dataframe_query(**args_exec)
            except Exception as e:
                res = {"error": f"dataframe_query failed: {e}"}

            context["last_rows"] = res.get("preview")
            if res.get("artifact_path"):
                artifacts.append(res["artifact_path"])

            calls.append({"tool": tool, "args": args_exec, "result_meta": {
                "rowcount": res.get("rowcount"),
                "artifact_path": res.get("artifact_path"),
                "sheet_used": res.get("sheet_used", sheet_used),
                "available_sheets": available[:20],
                "error": res.get("error"),
            }, "result": res})

        elif tool == "chart":
            rows = args.get("rows")
            if isinstance(rows, str) and rows.strip() == "$prev_rows":
                rows = context.get("last_rows") or []
            artifact_folder = args.get("artifact_folder") or _artifact_folder("charts", app_paths)

            args_exec = {
                "kind": args.get("kind", "bar"),
                "rows": rows,
                "x": args.get("x"),
                "y": args.get("y"),
                "title": args.get("title", "Chart"),
                "artifact_folder": artifact_folder,
                "base_name": args.get("base_name", "chart"),
            }
            try:
                res = chart(**args_exec)
            except Exception as e:
                res = {"error": f"chart failed: {e}"}

            if res.get("image_path"):
                artifacts.append(res["image_path"])
            calls.append({"tool": tool, "args": args_exec, "result_meta": {"image_path": res.get("image_path"), "error": res.get("error")}})

        elif tool == "kb_search":
            query = args.get("query")
            top_k = int(args.get("k", 5))
            try:
                res = kb_search(query or "", top_k)
            except Exception as e:
                res = {"error": f"kb_search failed: {e}", "citations": []}
            calls.append({"tool": tool, "args": {"query": query, "k": top_k}, "result_meta": {
                "citations": res.get("citations", []),
                "error": res.get("error")
            }})
            context["kb_chunks"] = res.get("chunks", [])
            context["kb_citations"] = res.get("citations", [])

        else:
            calls.append({"tool": tool, "args": args, "error": "Unsupported tool"})

    return {"calls": calls, "context": context, "artifacts": artifacts}


def _pick_fallback_sheet(available_sheets: List[str], requested: Optional[str]) -> Optional[str]:
    if not available_sheets:
        return None
    if requested in available_sheets:
        return requested
    lower = [s.lower() for s in available_sheets]
    for key in ("wip", "aged wip", "g512", "work in progress"):
        for i, s in enumerate(lower):
            if key in s:
                return available_sheets[i]
    for key in ("inventory", "finished goods", "fg"):
        for i, s in enumerate(lower):
            if key in s:
                return available_sheets[i]
    return available_sheets[0]


# =============================================================================
# Quant/KB merging for final answer
# =============================================================================
def _summarize_exec(exec_result: Dict[str, Any]) -> Tuple[str, List[str]]:
    calls = exec_result["calls"]
    artifacts = exec_result["artifacts"]

    lines = []
    for c in calls:
        if c.get("error"):
            lines.append(f"Tool error: {c['error']}")
        elif c["tool"] == "dataframe_query":
            meta = c.get("result_meta", {})
            result = c.get("result", {})
            
            # Basic info
            lines.append(f"Dataframe Query Results:")
            lines.append(f"• Sheet: {meta.get('sheet_used')}")
            lines.append(f"• Total rows: {meta.get('rowcount','?')}")
            
            # Include actual data preview for LLM context
            preview_data = result.get("preview", [])
            if preview_data:
                lines.append(f"• Top results found:")
                
                # Convert preview to readable format
                for i, row in enumerate(preview_data[:10]):  # Limit to top 10 for context
                    if i == 0:
                        # Show column headers
                        lines.append(f"   Row {i+1}: {dict(row)}")
                    else:
                        # Show subsequent rows more compactly
                        key_fields = {}
                        for k, v in row.items():
                            if k in ['Part No', 'Part Description', 'Age', 'Total Cost'] or 'age' in k.lower():
                                key_fields[k] = v
                        if key_fields:
                            lines.append(f"   Row {i+1}: {key_fields}")
                        elif len(str(row)) < 200:  # Show full row if short
                            lines.append(f"   Row {i+1}: {dict(row)}")
                        else:  # Truncate very long rows
                            lines.append(f"   Row {i+1}: [data truncated - {len(row)} fields]")
            
            if meta.get("artifact_path"):
                lines.append(f"• Full results saved to: {meta.get('artifact_path')}")
                
        elif c["tool"] == "chart":
            meta = c.get("result_meta", {})
            lines.append(f"Chart Generated • type={meta.get('chart_type','?')} • saved={meta.get('chart_path','?')}")
        else:
            # Other tools
            lines.append(f"Tool: {c['tool']} executed")
            
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


# =============================================================================
# Intent classification (robust + external-friendly)
# =============================================================================
SUPPORTED_INTENTS = [
    "compare", "root_cause", "forecast", "summarize",
    "eda", "rank", "anomaly", "optimize", "filter",
]

WIP_EO_BIAS_KEYWORDS = {
    "wip": ["wip", "work in progress", "job", "shop order", "wo ", "work order"],
    "eo": ["e&o", "excess", "obsolete", "slow mover", "dead stock", "write-off", "write off"],
}

DEFAULT_INTENT_MODEL = os.getenv("INTENT_MODEL", "gpt-4o-mini")
CONFIDENCE_THRESHOLD = 0.52


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
    # Use your external classifier if present
    if _external_classify_intent is not None:
        try:
            out = _external_classify_intent(user_question)
            out["model_size"] = out.get("model_size") or ("large" if out.get("intent") in {"root_cause", "forecast"} else "small")
            return out
        except Exception:
            pass

    if client is None:
        client = get_openai_client()
    prompt = _build_intent_prompt(user_question)

    try:
        resp = client.responses.create(model=DEFAULT_INTENT_MODEL, input=[{"role": "user", "content": prompt}], temperature=0.0)
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


