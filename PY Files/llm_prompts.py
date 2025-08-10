# PY Files/llm/prompts.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import textwrap

# -----------------------------
# High-level: prompt scaffold
# -----------------------------

SYSTEM_ROLE = """\
You are SCIE Ethos — an enterprise supply-chain analyst copilot.
You must produce accurate, auditable, and action-oriented answers grounded in data and cited sources.
Operate CLOUD-ONLY (Dropbox paths and cloud services). Never read or write local files.

Core behaviors:
- Use the tool layer for ALL data operations, comparisons, charts, and saves.
- Prefer numeric evidence from cleansed data, then enrich with KB guidance.
- Be explicit with units, filters, time windows, and assumptions.
- When uncertain, ask a brief clarifying question OR state uncertainty with options.
- Keep answers concise at the top, with expandable detail below.
- Always include citations to data artifacts and KB sources when used.
"""

# Default domain glossary (can be overridden by caller)
DEFAULT_GLOSSARY = """\
MRB = Material Review Board (nonconforming or held inventory pending disposition)
MRP = Material Requirements Planning
WIP = Work in Progress (incomplete jobs/work orders)
FG = Finished Goods
E&O = Excess & Obsolete Inventory (low/zero velocity items; write-down risk)
ROP = Reorder Point, EOQ = Economic Order Quantity, SS = Safety Stock
Usage-to-Stock Ratio = trailing usage / on-hand inventory (by value or qty)
Aging Buckets = time-in-state breakdown (e.g., 0–30, 31–60 days, etc.)
"""

# Answer style presets
ANSWER_STYLE_CONCISE = "concise"
ANSWER_STYLE_DETAILED = "detailed"


def _render_tool_guide(tools: Dict[str, Dict[str, Any]]) -> str:
    """
    Render a compact tool directory for the model.
    tools: { tool_name: { "purpose": str, "args_schema": dict, "returns": str } }
    """
    lines = ["You can call the following tools. Choose only those needed:"]
    for name, spec in tools.items():
        purpose = spec.get("purpose", "").strip()
        returns = spec.get("returns", "").strip()
        args = spec.get("args_schema", {})
        schema_str = json.dumps(args, ensure_ascii=False)
        lines.append(f"- {name}: {purpose}\n  args_schema: {schema_str}\n  returns: {returns}")
    return "\n".join(lines)


def _render_retrieval_context(
    quantitative_context: Optional[str],
    kb_context: Optional[str],
    matched_artifacts: Optional[List[str]] = None,
    matched_docs: Optional[List[str]] = None,
) -> str:
    """
    Quantitative context: summaries, metrics, small tables (text), paths to cleansed artifacts.
    KB context: short extracts/snippets relevant to the question.
    """
    parts = []
    if quantitative_context:
        parts.append("=== Quantitative Findings ===")
        parts.append(quantitative_context.strip())
    if matched_artifacts:
        parts.append("Artifacts (data citations):")
        for p in matched_artifacts:
            parts.append(f"- {p}")
    if kb_context:
        parts.append("\n=== Knowledgebase Excerpts ===")
        parts.append(kb_context.strip())
    if matched_docs:
        parts.append("KB Sources (citations):")
        for d in matched_docs:
            parts.append(f"- {d}")
    return "\n".join(parts).strip()


def _render_stepwise_plan(intent: str) -> str:
    steps_by_intent = {
        "compare": [
            "Confirm filters/time windows and comparison keys.",
            "Call dataframe_query to aggregate and align metrics.",
            "Call chart if visuals help (save under 02_EDA_Charts).",
            "Summarize top differences and likely drivers.",
        ],
        "root_cause": [
            "Confirm scope (file(s), period, product families, regions).",
            "Call root_cause to compute drivers and anomalies.",
            "Enrich with KB guidance on mitigation actions.",
            "Return ranked drivers with recommended actions.",
        ],
        "forecast": [
            "Confirm target (demand, SS, ROP, E&O risk horizon).",
            "Call forecast with required parameters and history window.",
            "Explain assumptions; show key parameters and outputs.",
        ],
        "eda": [
            "Call dataframe_query for distributions/rollups.",
            "Call chart for key plots; describe notable patterns.",
        ],
        "rank": [
            "Define ranking metric clearly (e.g., E&O by value).",
            "Call dataframe_query to compute scores and top-N.",
        ],
        "kb_lookup": [
            "Summarize the question to a targeted KB query.",
            "Call kb_search; extract applicable guidance.",
        ],
        "anomaly": [
            "Call dataframe_query to compute baseline and z-scores.",
            "Highlight outliers and possible data issues.",
        ],
        "optimize": [
            "Clarify objective and constraints.",
            "Propose actions and tradeoffs grounded in data.",
        ],
        "filter": [
            "Confirm filters and expected output shape.",
            "Return a compact table preview + path to full artifact.",
        ],
    }
    steps = steps_by_intent.get(intent, ["Plan the minimal set of tool calls to answer accurately."])
    bullets = "\n".join(f"- {s}" for s in steps)
    return f"Planned Steps for intent='{intent}':\n{bullets}"


def _render_answer_format(style: str) -> str:
    return textwrap.dedent(f"""\
    Answer Format ({style}):
    1) TL;DR — one paragraph with the direct answer.
    2) What I did — list of tools called and key filters/assumptions.
    3) Findings — concise bullets with numbers and units.
    4) Recommendations — concrete next actions (who/what/when).
    5) Citations — data artifacts and KB sources used (paths/IDs).
    If low confidence: include a brief clarifying question.

    ALWAYS:
    - Use concrete dates and units.
    - State uncertainty and data gaps when relevant.
    - Provide Dropbox paths for any saved outputs you reference.
    """)


def _render_governance_block() -> str:
    return textwrap.dedent("""\
    Governance & Safety:
    - Do not fabricate data or paths. If unknown, say so plainly.
    - Never expose secrets. Redact PII (names, emails, phone numbers) in outputs.
    - Abstain and ask a clarifying question when the query is underspecified.
    - Validate that narrative matches numbers (sanity check). If mismatch, flag it.
    """)


def build_scaffold_messages(
    user_question: str,
    *,
    intent: str,
    tools_catalog: Dict[str, Dict[str, Any]],
    quantitative_context: Optional[str] = None,
    kb_context: Optional[str] = None,
    matched_artifacts: Optional[List[str]] = None,
    matched_docs: Optional[List[str]] = None,
    glossary: Optional[str] = None,
    answer_style: str = ANSWER_STYLE_CONCISE,
    model_size: str = "auto",   # "small" | "large" | "auto"
) -> List[Dict[str, str]]:
    """
    Build the messages array for an LLM call, with all scaffold layers.
    You still decide model routing outside (small vs large).
    """
    sys = SYSTEM_ROLE.strip()
    glos = (glossary or DEFAULT_GLOSSARY).strip()
    tool_guide = _render_tool_guide(tools_catalog)
    retrieval = _render_retrieval_context(
        quantitative_context=quantitative_context,
        kb_context=kb_context,
        matched_artifacts=matched_artifacts,
        matched_docs=matched_docs,
    )
    plan = _render_stepwise_plan(intent=intent)
    fmt = _render_answer_format(answer_style)
    gov = _render_governance_block()

    system_block = "\n\n".join([sys, "Domain Glossary:\n" + glos, "Tool Guide:\n" + tool_guide, gov])

    user_block = textwrap.dedent(f"""\
    USER QUESTION:
    {user_question.strip()}

    CONTEXT (if any):
    {retrieval if retrieval else "(none)"}

    {plan}

    Follow the Answer Format exactly.
    """)

    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block},
    ]


# Optional helper: minimal catalog for quick starts
def minimal_tools_catalog() -> Dict[str, Dict[str, Any]]:
    """
    Returns a minimal example of tool specs.
    Replace with your live registry from tools_runtime when wired.
    """
    return {
        "dataframe_query": {
            "purpose": "Filter, group, aggregate, and join cleansed sheets; returns preview + saved artifact path.",
            "args_schema": {
                "file_refs": ["dropbox://04_Data/01_Cleansed_Files/..."],
                "ops": [{"op": "groupby", "by": ["plant"], "metrics": [{"col": "extended_cost", "agg": "sum"}]}],
                "filters": [{"col": "period", "op": "in", "value": ["Q1","Q2"]}],
                "limit": 50
            },
            "returns": "dict with 'preview' (rows) and 'artifact_path' (Dropbox path to CSV/Parquet)."
        },
        "chart": {
            "purpose": "Create a chart and save PNG under 04_Data/02_EDA_Charts in Dropbox.",
            "args_schema": {
                "kind": "bar|line|scatter",
                "data_ref": "artifact path or inline rows",
                "x": "column",
                "y": ["columns"],
                "title": "string"
            },
            "returns": "dict with 'image_path' (Dropbox path) and 'details'."
        },
        "kb_search": {
            "purpose": "Retrieve k relevant knowledgebase chunks for enrichment.",
            "args_schema": {"query": "string", "k": 5},
            "returns": "dict with 'chunks' and 'citations' (file IDs/paths)."
        }
    }
