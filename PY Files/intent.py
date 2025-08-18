# PY Files/intent.py
from __future__ import annotations
from typing import Dict, Tuple
import re

from llm_client import get_openai_client, chat_completion

# Allowed intents (keep in sync with orchestrator/tools and instructions_master.yaml)
ALLOWED_INTENTS = [
    "compare",         # compare time windows/segments/regions
    "root_cause",      # explain drivers and anomalies (RCA)
    "forecast",        # demand/SS/ROP/E&O projections
    "movement_analysis", # compare movement between periods (Q1→Q2)
    "eda",             # explore data distributions/rollups
    "rank",            # top-N by a metric
    "kb_lookup",       # guidance/policy/SOP lookup
    "anomaly",         # detect outliers/spikes
    "optimize",        # propose actions under constraints
    "filter",          # retrieve a filtered table preview
    "eo_analysis",     # Excess & Obsolete analysis
    "scenario",        # what-if analysis
    "exec_summary",    # executive summaries
    "gap_check",       # identify missing data
]

# Lightweight, cheap regex/keyword mapping
KEYMAP = [
    (r"\b(root cause|why|driver|explain|due to|cause of|reason)\b", "root_cause"),
    (r"\bforecast|predict|projection|expected|plan|plan(?:ning)?\b", "forecast"),
    (r"\b(movement|trend|shift|aging|bucket|q1|q2|quarter|period)\b", "movement_analysis"),
    (r"\bcompare|vs\.|versus|delta|change|difference\b", "compare"),
    (r"\btop\s*\d+|rank|ranking|highest|lowest|top[-\s]?n\b", "rank"),
    (r"\banomal(y|ies)|outlier|spike|unexpected|deviation\b", "anomaly"),
    (r"\boptimi[sz]e|recommendation|trade[-\s]?off|action plan|what should we do\b", "optimize"),
    (r"\bfilter|where|only.*rows|subset|slice\b", "filter"),
    (r"\bpolicy|SOP|guidance|how do we|best practice|kb\b", "kb_lookup"),
    (r"\bdistribution|histogram|profile|breakdown|EDA\b", "eda"),
    (r"\b(excess|obsolete|E&O|slow moving|write.?down)\b", "eo_analysis"),
    (r"\b(scenario|what.?if|simulation|assumption)\b", "scenario"),
    (r"\b(executive|summary|overview|kpi|recap)\b", "exec_summary"),
    (r"\b(missing|gap|incomplete|coverage)\b", "gap_check"),
]

# Intents that typically need the larger model for reasoning
LARGE_MODEL_INTENTS = {"root_cause", "forecast", "movement_analysis", "optimize", "scenario", "eo_analysis"}


def _cheap_rules(question: str) -> Tuple[str, float, str]:
    """Return (intent, confidence, reason) using regex/keywords."""
    q = (question or "").strip().lower()
    if not q:
        return "eda", 0.1, "Empty question; defaulting to exploratory data analysis."
    for pat, intent in KEYMAP:
        if re.search(pat, q):
            return intent, 0.65, f"Matched keyword pattern: {pat}"
    # Soft defaults
    if any(w in q for w in ["show", "list", "table", "summary"]):
        return "filter", 0.5, "Generic retrieval phrasing suggests filter/preview."
    if any(w in q for w in ["trend", "over time", "by month", "by plant", "group by"]):
        return "compare", 0.5, "Comparative phrasing suggests compare."
    return "eda", 0.4, "No strong cues; defaulting to EDA."


def classify_intent(question: str) -> Dict[str, str]:
    """
    Classify a user question into one of ALLOWED_INTENTS.
    Strategy: fast rules -> GPT disambiguation (constrained).
    Returns: {intent, reason, confidence (str), model_size}
    """
    # 1) Cheap heuristic pass
    intent, conf, why = _cheap_rules(question)

    # 2) If confidence is modest, ask GPT to choose among allowed labels
    if conf < 0.8:
        try:
            client = get_openai_client()
            labels = ", ".join([f"'{x}'" for x in ALLOWED_INTENTS])
            prompt = (
                "Classify the user's question into ONE of these intents exactly:\n"
                f"[{labels}]\n\n"
                "Rules:\n"
                "- Return only the label.\n"
                "- If the question is purely about company policy/SOPs or 'how to', use 'kb_lookup'.\n"
                "- If it's about explaining why a metric changed, use 'root_cause'.\n"
                "- If it's about future values or parameters (SS/ROP/E&O), use 'forecast'.\n"
                "- If it's about comparing movement between periods (Q1→Q2), use 'movement_analysis'.\n"
                "- If it's about excess/obsolete inventory analysis, use 'eo_analysis'.\n"
                "- If it's about what-if scenarios or simulations, use 'scenario'.\n"
                "- If it's asking for executive summary or KPI recap, use 'exec_summary'.\n"
                "- If it's asking to rank or top-N, use 'rank'.\n"
                "- If it's asking for a filtered subset or table preview, use 'filter'.\n"
                "- If it's about identifying missing data gaps, use 'gap_check'.\n"
                "- If none of the above, choose 'eda'.\n\n"
                f"User question: {question!r}"
            )
            messages = [
                {"role": "system", "content": "You are a precise intent classifier for a supply-chain analyst copilot."},
                {"role": "user", "content": prompt},
            ]
            label = (chat_completion(client, messages, model="gpt-4o-mini") or "").strip().lower().strip(" '\"\n")
            if label in ALLOWED_INTENTS:
                intent = label
                why = f"GPT disambiguation selected '{label}'."
                conf = max(conf, 0.85)
            else:
                why = f"GPT returned '{label}', falling back to heuristic intent '{intent}'."
        except Exception as e:
            why = f"{why} (GPT fallback unavailable: {e})"

    model_size = "large" if intent in LARGE_MODEL_INTENTS else "small"
    return {
        "intent": intent,
        "reason": why,
        "confidence": f"{conf:.2f}",
        "model_size": model_size,
    }
