# chat_ui.py — Universal KB Summaries Q&A (no special cases), deep-read default ON
# Compatible with existing main.py that calls `render_chat_assistant()`.
# This module does NOT touch DP mode or your other orchestration.

import os
import re
import json
import math
from pathlib import Path
from collections import Counter

import streamlit as st

# Optional deep-read libs (gracefully skipped if missing)
try:
    import pdfplumber  # PDF
except Exception:
    pdfplumber = None

try:
    import docx  # python-docx (DOCX)
except Exception:
    docx = None

try:
    import extract_msg  # Outlook .msg
except Exception:
    extract_msg = None


# --------------------------------------------------------------------------------------
# THEME-SAFE STYLING (no set_page_config; respect your config.toml)
# --------------------------------------------------------------------------------------
bg  = st.get_option("theme.backgroundColor") or "#0e1117"
sb  = st.get_option("theme.secondaryBackgroundColor") or "#1b1f24"
tx  = st.get_option("theme.textColor") or "#e6edf3"
pc  = st.get_option("theme.primaryColor") or "#8ab4f8"

st.markdown(f"""
<style>
  .card {{
      background: {sb};
      border: 1px solid rgba(255,255,255,0.08);
      color: {tx};
      border-radius: 10px;
      padding: 12px 14px;
      margin: 8px 0;
  }}
  .pill {{
      display:inline-block; padding:3px 8px; border-radius:12px;
      border:1px solid rgba(255,255,255,0.2); color:{tx};
      background: rgba(138,180,248,0.15);
      margin-right:6px; font-size:0.8rem;
  }}
  .cite {{ color:{pc}; font-size:0.85rem; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------
def _detect_project_root() -> str:
    pr = os.getenv("PROJECT_ROOT")
    if pr:
        return pr.rstrip("/")
    for c in ["/Project_Root", "/Apps/Ethos LLM/Project_Root"]:
        if Path(c).exists():
            return c
    return "."

PROJECT_ROOT      = _detect_project_root()
KB_ROOT           = Path(PROJECT_ROOT) / "06_LLM_Knowledge_Base"
KB_SUMMARIES_DIR  = KB_ROOT / "KB_Summaries"
KB_MEETING_DIR    = KB_ROOT / "Meeting Minutes"
KB_EMAIL_DIR      = KB_ROOT / "Email"
OPTIONAL_SUMS_04  = Path(PROJECT_ROOT) / "04_Data" / "03_Summaries"


# --------------------------------------------------------------------------------------
# SUMMARIES ENGINE: INDEX, SEARCH, COMPOSE
# --------------------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z0-9]+")

def _tokenize(text: str):
    return [w.lower() for w in _WORD.findall(text or "")]

def _bm25_score(query_tokens, doc_tokens, avgdl, k1=1.5, b=0.75):
    tf = Counter(doc_tokens)
    score = 0.0
    dl = len(doc_tokens)
    for qt, idf in query_tokens:  # query_tokens: list of (token, idf)
        f = tf.get(qt, 0)
        if f == 0:
            continue
        score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl)))
    return score

def _load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _yield_summary_docs(root_dir: Path):
    if not root_dir.exists():
        return
    for p in root_dir.rglob("*_summary.json"):
        js = _load_json(p)
        if not js or not isinstance(js, dict):
            continue

        # Build a searchable string from all string-like content inside the JSON
        text_fields = []
        def collect(o):
            if isinstance(o, str):
                text_fields.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    collect(v)
            elif isinstance(o, list):
                for v in o:
                    collect(v)
        collect(js)
        searchable = "\n".join(text_fields)

        yield {
            "path": p,
            "dir": p.parent,
            "file_name": js.get("file_name") or p.name,
            "document_type": js.get("document_type", ""),
            "summary_json": js,
            "search_text": searchable
        }

@st.cache_resource(show_spinner=False)
def _build_index(include_summaries_04: bool):
    sources = [KB_SUMMARIES_DIR]
    if include_summaries_04 and OPTIONAL_SUMS_04.exists():
        sources.append(OPTIONAL_SUMS_04)

    docs = []
    for src in sources:
        docs.extend(list(_yield_summary_docs(src)))

    tokenized_docs = []
    df = Counter()
    for d in docs:
        toks = _tokenize(d["search_text"])
        tokenized_docs.append(toks)
        df.update(set(toks))

    N = len(docs)
    avgdl = (sum(len(t) for t in tokenized_docs) / N) if N else 1
    idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}

    return {"docs": docs, "tokens": tokenized_docs, "idf": idf, "avgdl": avgdl, "N": N}

def _search_summaries(index, query: str, topk=6):
    if index["N"] == 0:
        return []
    qtoks_raw = _tokenize(query)
    if not qtoks_raw:
        return []
    query_tokens = [(t, index["idf"].get(t, 0.0)) for t in qtoks_raw]
    scores = []
    for i, toks in enumerate(index["tokens"]):
        s = _bm25_score(query_tokens, toks, index["avgdl"])
        scores.append((s, i))
    scores.sort(reverse=True)
    return [(index["docs"][i], s) for s, i in scores[:topk] if s > 0]


# --------------------------------------------------------------------------------------
# DEEP READ (DRILL-DOWN): PDFs, DOCX, MSG
# --------------------------------------------------------------------------------------
def _deep_read_docx(path: Path, max_chars=3000) -> str:
    if docx is None or not path.exists():
        return ""
    try:
        d = docx.Document(str(path))
        text = "\n".join(p.text for p in d.paragraphs if p.text)
        return text[:max_chars]
    except Exception:
        return ""

def _deep_read_pdf(path: Path, max_chars=3000) -> str:
    if pdfplumber is None or not path.exists():
        return ""
    out = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages[:8]:
                out.append(page.extract_text() or "")
        return "\n".join(out)[:max_chars]
    except Exception:
        return ""

def _deep_read_msg(path: Path, max_chars=3000) -> str:
    if extract_msg is None or not path.exists():
        return ""
    try:
        m = extract_msg.Message(str(path))
        body = (m.body or "")
        subj = (m.subject or "")
        return (subj + "\n\n" + body)[:max_chars]
    except Exception:
        return ""

def _enrich_with_deep_read(summary_json: dict) -> str:
    """
    Priority 1: Use summary_json['sources'] (relative to KB root).
    Priority 2: If absent, stem-match 'file_name' under Meeting Minutes & Email.
    """
    texts = []

    # 1) explicit sources list (recommended)
    for key in ("sources", "source_files"):
        srcs = summary_json.get(key, [])
        if isinstance(srcs, list):
            for rel in srcs:
                p = (KB_ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
                suf = p.suffix.lower()
                if suf == ".pdf":
                    texts.append(_deep_read_pdf(p))
                elif suf == ".docx":
                    texts.append(_deep_read_docx(p))
                elif suf == ".msg":
                    texts.append(_deep_read_msg(p))
    if any(texts):
        return "\n".join(t for t in texts if t)

    # 2) stem-based guess
    stem = Path(summary_json.get("file_name", "")).stem
    for folder in (KB_MEETING_DIR, KB_EMAIL_DIR, KB_ROOT):
        for gp in (list(folder.rglob(f"{stem}*.pdf"))[:1] +
                   list(folder.rglob(f"{stem}*.docx"))[:1] +
                   list(folder.rglob(f"{stem}*.msg"))[:1]):
            suf = gp.suffix.lower()
            if suf == ".pdf":
                return _deep_read_pdf(gp)
            if suf == ".docx":
                return _deep_read_docx(gp)
            if suf == ".msg":
                return _deep_read_msg(gp)
    return ""


# --------------------------------------------------------------------------------------
# COMPOSE CONSOLIDATED ANSWER (no special-cases)
# --------------------------------------------------------------------------------------
def _compose_answer(query: str, hits, deep_read: bool):
    """
    Build a consolidated answer from the best summaries.
    Pulls participants, actions, decisions, launch tables, frame summaries,
    and optionally short excerpts from source docs.
    """
    if not hits:
        return "I couldn’t find any summaries relevant to this question."

    blocks = []
    for d, score in hits:
        sj = d["summary_json"]
        parts = []

        # High-signal fields first
        participants = sj.get("participants")
        if participants:
            parts.append("**Participants:** " + ", ".join(participants))

        decisions = sj.get("decisions")
        if decisions:
            parts.append("**Decisions:** " + json.dumps(decisions, ensure_ascii=False))

        actions = sj.get("actions")
        if actions:
            act_lines = []
            for a in actions[:10]:
                if isinstance(a, dict):
                    who = a.get("owner")
                    item = a.get("item") or a.get("action") or ""
                    due = a.get("due")
                    line = f"- {item}"
                    if who: line = f"- ({who}) {item}"
                    if due: line += f" — due {due}"
                    act_lines.append(line)
                elif isinstance(a, str):
                    act_lines.append("- " + a)
            if act_lines:
                parts.append("**Actions:**\n" + "\n".join(act_lines))

        # Launch table(s)
        lt = sj.get("launch_table") or sj.get("launch_table_manufactured") or sj.get("launch_table_traded") or []
        if lt:
            rows = []
            for r in lt[:12]:
                rows.append(
                    f"{r.get('frame','')} {r.get('component','')}: "
                    f"{r.get('decision','')} x{r.get('sets','')}"
                )
            parts.append("**Launches:** " + "; ".join(rows))

        # Frame summary
        fs = sj.get("frame_summary")
        if fs:
            fs_text = "; ".join(f"{k}: {v}" for k, v in fs.items())
            parts.append("**Frame Summary:** " + fs_text)

        # Financial / cash-flow if present
        for money_key in ("cash_flow", "financials"):
            cf = sj.get(money_key)
            if cf:
                parts.append(f"**{money_key.replace('_',' ').title()}:** " + json.dumps(cf, ensure_ascii=False))

        # Narrative summary
        summ = sj.get("summary")
        if summ:
            parts.append("**Summary:** " + summ)

        # Optional deep-read excerpt (default ON from UI)
        if deep_read:
            extra = _enrich_with_deep_read(sj)
            if extra:
                parts.append("**Source Excerpt:** " + extra[:1000])

        if parts:
            block = (
                f"<div class='card'>"
                f"<div class='pill'>{sj.get('document_type','Summary')}</div> "
                f"<div class='pill'>{d['file_name']}</div><br/>"
                + "<br/>".join(parts) +
                f"</div>"
            )
            blocks.append(block)

    return ("Here’s what the knowledge-base summaries show (ranked):\n\n" +
            "\n\n".join(blocks) if blocks else
            "I found matching summaries but none had extractable content.")


# --------------------------------------------------------------------------------------
# PUBLIC ENTRYPOINT (expected by main.py)
# --------------------------------------------------------------------------------------
def render_chat_assistant():
    """
    This is the entrypoint main.py imports and calls.
    It renders a complete Q&A UI over KB Summaries (and optional 04_Data summaries),
    with deep-read default ON. It does NOT interfere with DP mode or any other flows.
    """
    st.header("🔎 KB Summaries — Enterprise Q&A")

    with st.sidebar:
        st.subheader("Search Settings")
        include_04 = st.checkbox("Also search /04_Data/03_Summaries", value=False)
        deep_read  = st.checkbox("Deep-read source docs (Meeting Minutes / Email)", value=True)
        st.caption(f"Project Root: {PROJECT_ROOT}")
        st.caption(f"KB Summaries: {str(KB_SUMMARIES_DIR)}")
        st.caption(f"Meeting Minutes: {str(KB_MEETING_DIR)}")
        st.caption(f"Email: {str(KB_EMAIL_DIR)}")
        if include_04:
            st.caption(f"04_Data Summaries: {str(OPTIONAL_SUMS_04)}")

    # Build index (cached)
    index = _build_index(include_04)

    # Input
    query = st.text_input(
        "Ask a question (e.g., 'what did we launch in 2024 Q2 for F7?', 'cash approvals for Dec 2024 launch', 'list participants for last SIOP')",
        placeholder="Type your question here…"
    )

    # Action
    if st.button("Search") or query:
        with st.spinner("Searching KB Summaries…"):
            hits = _search_summaries(index, query, topk=8)
            answer = _compose_answer(query, hits, deep_read=deep_read)

        st.markdown(answer, unsafe_allow_html=True)

        # Debug panel
        with st.expander("Debug • Ranked Matches"):
            if not hits:
                st.write("No matches.")
            else:
                rows = []
                for d, s in hits:
                    rows.append({
                        "score": round(s, 3),
                        "file_name": d["file_name"],
                        "doc_type": d["document_type"],
                        "path": str(d["path"]),
                    })
                st.dataframe(rows, use_container_width=True)
