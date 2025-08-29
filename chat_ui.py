# chat_ui.py — Universal KB Summaries QA (no special-cases), deep-read default ON
# Searches /Project_Root/06_LLM_Knowledge_Base/KB_Summaries and optionally /04_Data/03_Summaries,
# ranks best summaries, composes one consolidated answer with citations,
# and (by default) drills down into Meeting Minutes / Email source docs referenced by each summary.
# This file is self-contained and does NOT touch DP mode or your other app modules.

import os
import re
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import streamlit as st

# Optional deep-read libs (gracefully skipped if not installed)
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
# THEME-SAFE UI (no hardcoded light colors)
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="KB Summaries Q&A", layout="wide")

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
  .small {{ opacity:0.8; font-size:0.9rem; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------------
def detect_project_root() -> str:
    pr = os.getenv("PROJECT_ROOT")
    if pr:
        return pr.rstrip("/")
    for c in ["/Project_Root", "/Apps/Ethos LLM/Project_Root"]:
        if Path(c).exists():
            return c
    return "."

PROJECT_ROOT = detect_project_root()

KB_ROOT           = Path(PROJECT_ROOT) / "06_LLM_Knowledge_Base"
KB_SUMMARIES_DIR  = KB_ROOT / "KB_Summaries"
KB_MEETING_DIR    = KB_ROOT / "Meeting Minutes"
KB_EMAIL_DIR      = KB_ROOT / "Email"
OPTIONAL_SUMS_04  = Path(PROJECT_ROOT) / "04_Data" / "03_Summaries"


# --------------------------------------------------------------------------------------
# SUMMARIES ENGINE: INDEX, SEARCH, COMPOSE
# --------------------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str):
    return [w.lower() for w in _WORD.findall(text or "")]

def bm25_score(query_tokens, doc_tokens, avgdl, k1=1.5, b=0.75):
    tf = Counter(doc_tokens)
    score = 0.0
    dl = len(doc_tokens)
    for qt, idf in query_tokens:  # query_tokens is list of (token, idf)
        f = tf.get(qt, 0)
        if f == 0:
            continue
        score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl)))
    return score

def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def yield_summary_docs(root_dir: Path):
    if not root_dir.exists():
        return
    for p in root_dir.rglob("*_summary.json"):
        js = load_json(p)
        if not js or not isinstance(js, dict):
            continue

        # Build a searchable string from all string content inside the JSON
        text_fields = []
        def collect(obj):
            if isinstance(obj, str):
                text_fields.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, list):
                for v in obj:
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
def build_index(include_summaries_04: bool):
    sources = [KB_SUMMARIES_DIR]
    if include_summaries_04 and OPTIONAL_SUMS_04.exists():
        sources.append(OPTIONAL_SUMS_04)

    docs = []
    for src in sources:
        docs.extend(list(yield_summary_docs(src)))

    # Tokenize & compute IDF
    tokenized_docs = []
    df = Counter()
    for d in docs:
        toks = tokenize(d["search_text"])
        tokenized_docs.append(toks)
        df.update(set(toks))

    N = len(docs)
    avgdl = (sum(len(t) for t in tokenized_docs) / N) if N else 1
    idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}

    return {"docs": docs, "tokens": tokenized_docs, "idf": idf, "avgdl": avgdl, "N": N}

def search_summaries(index, query: str, topk=6):
    if index["N"] == 0:
        return []

    qtoks_raw = tokenize(query)
    if not qtoks_raw:
        return []

    query_tokens = [(t, index["idf"].get(t, 0.0)) for t in qtoks_raw]
    scores = []
    for i, toks in enumerate(index["tokens"]):
        s = bm25_score(query_tokens, toks, index["avgdl"])
        scores.append((s, i))

    scores.sort(reverse=True)
    hits = [(index["docs"][i], s) for s, i in scores[:topk] if s > 0]
    return hits


# --------------------------------------------------------------------------------------
# DEEP READ (DRILL-DOWN): PDFs, DOCX, MSG
# --------------------------------------------------------------------------------------
def deep_read_docx(path: Path, max_chars=3000) -> str:
    if docx is None or not path.exists():
        return ""
    try:
        d = docx.Document(str(path))
        text = "\n".join(p.text for p in d.paragraphs if p.text)
        return text[:max_chars]
    except Exception:
        return ""

def deep_read_pdf(path: Path, max_chars=3000) -> str:
    if pdfplumber is None or not path.exists():
        return ""
    out = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages[:8]:  # a few pages is enough for enrichment
                out.append(page.extract_text() or "")
        return "\n".join(out)[:max_chars]
    except Exception:
        return ""

def deep_read_msg(path: Path, max_chars=3000) -> str:
    if extract_msg is None or not path.exists():
        return ""
    try:
        m = extract_msg.Message(str(path))
        body = (m.body or "")
        subj = (m.subject or "")
        return (subj + "\n\n" + body)[:max_chars]
    except Exception:
        return ""

def enrich_with_deep_read(summary_json: dict) -> str:
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
                    texts.append(deep_read_pdf(p))
                elif suf == ".docx":
                    texts.append(deep_read_docx(p))
                elif suf == ".msg":
                    texts.append(deep_read_msg(p))
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
                return deep_read_pdf(gp)
            if suf == ".docx":
                return deep_read_docx(gp)
            if suf == ".msg":
                return deep_read_msg(gp)
    return ""


# --------------------------------------------------------------------------------------
# COMPOSE CONSOLIDATED ANSWER (no special-cases)
# --------------------------------------------------------------------------------------
def compose_answer(query: str, hits, deep_read: bool):
    """
    Build one consolidated answer from the best summaries.
    Pulls participants, actions, decisions, launch tables, frame summaries,
    and optionally short excerpts from source docs.
    """
    if not hits:
        return "I couldn’t find any summaries relevant to this question."

    blocks = []
    for d, score in hits:
        sj = d["summary_json"]
        parts = []

        # High-value fields first
        participants = sj.get("participants")
        if participants:
            parts.append("**Participants:** " + ", ".join(participants))

        # Decisions / actions
        decisions = sj.get("decisions")
        if decisions:
            parts.append("**Decisions:** " + json.dumps(decisions, ensure_ascii=False))

        actions = sj.get("actions")
        if actions:
            act_lines = []
            for a in actions[:8]:
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
            for r in lt[:10]:
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

        # Financial/cash-flow (if present)
        cf = sj.get("cash_flow") or sj.get("financials")
        if cf:
            cf_txt = json.dumps(cf, ensure_ascii=False)
            parts.append("**Cash/Financials:** " + cf_txt)

        # Narrative summary
        summ = sj.get("summary")
        if summ:
            parts.append("**Summary:** " + summ)

        # Optional deep-read excerpt
        if deep_read:
            extra = enrich_with_deep_read(sj)
            if extra:
                parts.append("**Source Excerpt:** " + extra[:800])

        # Wrap the block
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
# UI
# --------------------------------------------------------------------------------------
def run_summaries_qa_ui():
    st.title("🔎 KB Summaries — Enterprise Q&A")

    with st.sidebar:
        st.subheader("Search Settings")
        include_04 = st.checkbox("Also search /04_Data/03_Summaries", value=False)
        deep_read = st.checkbox("Deep-read source docs (Meeting Minutes / Email)", value=True)
        st.caption("KB Root: " + str(KB_ROOT))
        st.caption("KB Summaries: " + str(KB_SUMMARIES_DIR))
        st.caption("Meeting Minutes: " + str(KB_MEETING_DIR))
        st.caption("Email: " + str(KB_EMAIL_DIR))
        if include_04:
            st.caption("04_Data Summaries: " + str(OPTIONAL_SUMS_04))

    # Build index (cached)
    index = build_index(include_04)

    query = st.text_input(
        "Ask a question (e.g., 'what did we launch in 2024 Q2 for F7?', 'show cash approvals for Dec 2024 launch', 'list participants for last SIOP meeting')",
        placeholder="Type your question here…"
    )

    if st.button("Search") or query:
        with st.spinner("Searching KB Summaries…"):
            hits = search_summaries(index, query, topk=8)
            answer = compose_answer(query, hits, deep_read=deep_read)

        st.markdown(answer, unsafe_allow_html=True)

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


# Run the UI
if __name__ == "__main__":
    run_summaries_qa_ui()
