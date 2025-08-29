# chat_ui.py — Grounded KB Summaries + Drill-down (default ON)
# Exports render_chat_assistant() expected by main.py
# Layout preserved; no DP-mode changes.

import os, re, sys, json, time, math
from pathlib import Path
from typing import Optional
from collections import Counter

import streamlit as st

# Make local modules importable (unchanged)
sys.path.append(str((Path(__file__).resolve().parent / "PY Files").resolve()))

# Your existing modules (unchanged imports)
from constants import PROJECT_ROOT
from session import SessionState
from export_utils import ExportManager
from sources_drawer import SourcesDrawer
from data_needed_panel import DataNeededPanel
from assistant_bridge import run_query  # we will ground the model with evidence

# ------------------------ THEME SHIM (dark mode) ------------------------
_bg  = st.get_option("theme.backgroundColor") or "#0e1117"
_sb  = st.get_option("theme.secondaryBackgroundColor") or "#1b1f24"
_tx  = st.get_option("theme.textColor") or "#e6edf3"
_bd  = "rgba(255,255,255,0.12)"

st.markdown(f"""
<style>
:root {{ --secondary-background-color: {_sb}; --border-color: {_bd}; }}
.card {{ background: var(--secondary-background-color); border:1px solid var(--border-color);
        border-radius:10px; padding:12px 14px; margin:8px 0; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:8px;
         background: rgba(138,180,248,0.15); color:{_tx}; border:1px solid rgba(138,180,248,0.35);
         margin-right:6px; font-size:0.8rem; font-weight:600; }}
.stChatInput input {{ border-radius: 20px; }}
</style>
""", unsafe_allow_html=True)

# ------------------------ KB paths ------------------------
KB_ROOT      = Path(os.getenv("PROJECT_ROOT", PROJECT_ROOT)) / "06_LLM_Knowledge_Base"
KB_SUMS_DIR  = KB_ROOT / "KB_Summaries"
MM_DIR       = KB_ROOT / "Meeting Minutes"
EMAIL_DIR    = KB_ROOT / "Email"

# ------------------------ Optional deep-read libs ------------------------
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx
except Exception:
    docx = None
try:
    import extract_msg
except Exception:
    extract_msg = None

# ------------------------ Retrieval helpers ------------------------
_WORD = re.compile(r"[A-Za-z0-9]+")

def _tok(text: str):
    return [w.lower() for w in _WORD.findall(text or "")]

def _bm25(q, d, avgdl, k1=1.5, b=0.75):
    tf = Counter(d); dl = len(d); s = 0.0
    for t, idf in q:
        f = tf.get(t, 0)
        if f:
            s += idf * ((f*(k1+1)) / (f + k1*(1-b + b*dl/avgdl)))
    return s

def _load_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _collect_text(obj, out):
    if isinstance(obj, str): out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values(): _collect_text(v, out)
    elif isinstance(obj, list):
        for v in obj: _collect_text(v, out)

@st.cache_resource(show_spinner=False)
def _build_kb_index():
    docs=[]
    if KB_SUMS_DIR.exists():
        for p in KB_SUMS_DIR.rglob("*_summary.json"):
            js=_load_json(p)
            if not isinstance(js, dict): continue
            buf=[]; _collect_text(js, buf)
            docs.append({"path":p, "file_name": js.get("file_name") or p.name,
                         "doc_type": js.get("document_type",""), "js": js,
                         "text":"\n".join(buf)})
    # tokenize/IDF
    toks=[]; df=Counter()
    for d in docs:
        td=_tok(d["text"]); toks.append(td); df.update(set(td))
    N=len(docs); avgdl=(sum(len(t) for t in toks)/N) if N else 1
    idf={t: math.log((N-df[t]+0.5)/(df[t]+0.5)+1) for t in df}
    return {"docs":docs, "toks":toks, "idf":idf, "avgdl":avgdl, "N":N}

def _search_kb(index, query, topk=8):
    if index["N"]==0: return []
    qt=_tok(query); q=[(t, index["idf"].get(t,0.0)) for t in qt]
    scored=[]
    for i, dt in enumerate(index["toks"]):
        scored.append((_bm25(q, dt, index["avgdl"]), i))
    scored.sort(reverse=True)
    return [(index["docs"][i], s) for s,i in scored[:topk] if s>0]

# ------------------------ Drill-down (PDF/DOCX/MSG) ------------------------
def _deep_pdf(p: Path, max_chars=1200):
    if not (pdfplumber and p.exists()): return ""
    try:
        out=[]
        with pdfplumber.open(str(p)) as pdf:
            for pg in pdf.pages[:6]:
                out.append(pg.extract_text() or "")
        return "\n".join(out)[:max_chars]
    except Exception: return ""

def _deep_docx(p: Path, max_chars=1200):
    if not (docx and p.exists()): return ""
    try:
        d=docx.Document(str(p))
        return "\n".join(par.text for par in d.paragraphs if par.text)[:max_chars]
    except Exception: return ""

def _deep_msg(p: Path, max_chars=1200):
    if not (extract_msg and p.exists()): return ""
    try:
        m=extract_msg.Message(str(p))
        return ((m.subject or "") + "\n\n" + (m.body or ""))[:max_chars]
    except Exception: return ""

def _drill_sources(js: dict) -> str:
    """Priority 1: js['sources'] relative to KB root; Priority 2: stem-match 'file_name'."""
    texts=[]
    for key in ("sources","source_files"):
        arr=js.get(key,[])
        if isinstance(arr, list):
            for rel in arr:
                pp=(KB_ROOT/rel) if not Path(rel).is_absolute() else Path(rel)
                suf=pp.suffix.lower()
                if suf==".pdf": texts.append(_deep_pdf(pp))
                elif suf==".docx": texts.append(_deep_docx(pp))
                elif suf==".msg": texts.append(_deep_msg(pp))
    if texts: return "\n".join(t for t in texts if t)

    stem=Path(js.get("file_name","")).stem
    for folder in (MM_DIR, EMAIL_DIR, KB_ROOT):
        for gp in (list(folder.rglob(f"{stem}*.pdf"))[:1] +
                   list(folder.rglob(f"{stem}*.docx"))[:1] +
                   list(folder.rglob(f"{stem}*.msg"))[:1]):
            suf=gp.suffix.lower()
            if suf==".pdf": return _deep_pdf(gp)
            if suf==".docx": return _deep_docx(gp)
            if suf==".msg": return _deep_msg(gp)
    return ""

# ------------------------ Compose evidence + grounded summarization ------------------------
def _compose_evidence_cards(hits, include_excerpts=True) -> str:
    """Render evidence cards (summaries + optional excerpts)."""
    blocks=[]
    for d,_ in hits:
        js=d["js"]; parts=[]

        if js.get("participants"):
            parts.append("**Participants:** " + ", ".join(js["participants"]))
        if js.get("decisions"):
            parts.append("**Decisions:** " + json.dumps(js["decisions"], ensure_ascii=False))

        lt = js.get("launch_table") or js.get("launch_table_manufactured") or js.get("launch_table_traded") or []
        if lt:
            rows=[f"{r.get('frame','')} {r.get('component','')}: {r.get('decision','')} x{r.get('sets','')}" for r in lt[:10]]
            parts.append("**Launches:** " + "; ".join(rows))

        if js.get("frame_summary"):
            fs="; ".join(f"{k}: {v}" for k,v in js["frame_summary"].items())
            parts.append("**Frame Summary:** " + fs)

        if js.get("actions"):
            acts=[]
            for a in js["actions"][:10]:
                if isinstance(a, dict):
                    who=a.get("owner"); item=a.get("item") or a.get("action") or ""
                    due=a.get("due"); line=f"- {item}"
                    if who: line=f"- ({who}) {item}"
                    if due: line += f" — due {due}"
                    acts.append(line)
                elif isinstance(a,str):
                    acts.append("- " + a)
            if acts: parts.append("**Actions:**\n" + "\n".join(acts))

        if js.get("summary"):
            parts.append("**Summary:** " + js["summary"])

        if include_excerpts:
            extra=_drill_sources(js)
            if extra: parts.append("**Source Excerpt:** " + extra[:1200])

        if parts:
            blocks.append(
                f"<div class='card'><span class='badge'>{js.get('document_type','Summary')}</span> "
                f"<span class='badge'>{d['file_name']}</span><br/>" + "<br/>".join(parts) + "</div>"
            )
    return "\n\n".join(blocks)

def _build_grounded_prompt(user_q: str, hits) -> str:
    """Build a strict instruction + short context bundle so the model only uses evidence."""
    # cap how many summaries & how much excerpt we include to avoid huge prompts
    ctx = []
    for d,_ in hits[:4]:
        js = d["js"]
        piece = {
            "file": d["file_name"],
            "type": js.get("document_type",""),
            "participants": js.get("participants", []),
            "decisions": js.get("decisions", []),
            "launch_table": js.get("launch_table") or js.get("launch_table_manufactured") or js.get("launch_table_traded") or [],
            "frame_summary": js.get("frame_summary", {}),
            "summary": js.get("summary","")
        }
        excerpt = _drill_sources(js) or ""
        if excerpt:
            piece["excerpt"] = excerpt[:1200]
        ctx.append(piece)

    instruction = (
        "You are an enterprise analyst. Answer the question **using ONLY the provided context**. "
        "If the context is insufficient, say 'insufficient evidence'. "
        "Prefer precise facts (names, quantities, $ amounts, dates) and briefly cite the file names inline."
    )
    return instruction + "\n\nQuestion:\n" + user_q + "\n\nContext:\n" + json.dumps(ctx, ensure_ascii=False)

def grounded_answer_from_kb(user_q: str) -> Optional[str]:
    """Retrieve > build grounded prompt > call existing run_query with ONLY this context."""
    idx = _build_kb_index()
    hits = _search_kb(idx, user_q, topk=8)
    if not hits:
        return None

    # Compose visible evidence cards (we show alongside the summary)
    evidence_cards = _compose_evidence_cards(hits, include_excerpts=True)

    # Build grounded prompt for your existing model entrypoint (no hallucinations)
    grounded_prompt = _build_grounded_prompt(user_q, hits)

    # Call your existing model with the grounded prompt (no uploaded files passed here)
    # If your run_query signature differs, adapt as needed.
    try:
        resp = run_query(grounded_prompt)  # uses your assistant pipeline
        model_text = (resp or {}).get("answer", "") or (resp or {}).get("response", "")
    except Exception as e:
        model_text = ""

    # If model returns nothing, return just the evidence cards
    if not model_text or model_text.strip() == "":
        return "**📚 Knowledge Base Evidence**\n\n" + evidence_cards

    # Otherwise show the grounded model summary + evidence
    return "**📚 Knowledge Base Answer (Grounded)**\n\n" + model_text + "\n\n---\n\n" + evidence_cards

# ------------------------ UI (layout preserved) ------------------------
def render_chat_assistant():
    # session init
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "data_needed_panel" not in st.session_state:
        st.session_state.data_needed_panel = DataNeededPanel()

    st.title("🧠 SCIE Ethos LLM Assistant")
    st.caption("Grounded KB Summaries + Drill-down (no ungrounded answers)")

    # Sidebar (existing sections)
    with st.sidebar:
        st.header("🧠 SCIE Ethos Control Panel")
        st.info("💡 Dark mode is auto from theme. Drill-down is ON by default.")

        st.subheader("📎 Upload Documents")
        uploaded = st.file_uploader("Upload documents for analysis",
                                    type=['pdf','docx','pptx','txt','md','csv','xlsx'],
                                    accept_multiple_files=True)
        if uploaded:
            st.session_state.selected_files = uploaded
            st.success(f"✅ {len(uploaded)} file(s) uploaded")
            for f in uploaded: st.write(f"📄 {f.name}")
        else:
            st.session_state.selected_files = []

        st.markdown("---")
        st.subheader("📚 Knowledge Base Tools")
        if st.button("🔄 Index New Documents"):
            try:
                sys.path.append('PY Files')
                from kb_indexer import KBIndexer
                kb_path = os.getenv('KB_DBX_PATH', str(KB_ROOT))
                data_path = os.getenv('DATA_DBX_PATH', '/Project_Root/04_Data')
                res = KBIndexer(kb_path, data_path).process_new_files()
                st.success("✅ Indexing Complete")
                st.info(json.dumps(res, indent=2))
            except Exception as e:
                st.error(f"Indexing failed: {e}")

        st.markdown("---")
        st.subheader("📊 Data Needed & Gaps")
        st.session_state.data_needed_panel.load_from_session()
        st.session_state.data_needed_panel.render_panel(expanded=False)

    # Display chat history
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Pinned chat input
    if user_q := st.chat_input("Ask from your KB (e.g., launches, approvals, participants, org, cash model)…"):
        st.session_state.chat_messages.append({"role":"user","content":user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                # Grounded, data-only answer sourced from KB summaries + drill-down
                grounded = grounded_answer_from_kb(user_q)

                if not grounded:
                    placeholder.warning("No KB summaries matched this question. Try a different phrasing or index more documents.")
                else:
                    placeholder.markdown(grounded, unsafe_allow_html=True)

                # Save the assistant turn
                st.session_state.chat_messages.append({
                    "role":"assistant",
                    "content": grounded or "No KB evidence found."
                })

            except Exception as e:
                placeholder.error(f"Error: {e}")
                st.session_state.chat_messages.append({
                    "role":"assistant", "content": f"Error: {e}", "error": True
                })

    # Export block (minimal; you can extend with your original ExportManager config)
    if st.session_state.chat_messages:
        st.markdown("---")
        st.header("📤 Export")
        exp = ExportManager(0.95)
        data = {
            "messages": st.session_state.chat_messages,
            "exported_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧾 JSON"):
                try:
                    j = exp.export_to_json(data)
                    st.download_button("Download JSON", j, file_name=f"chat_{int(time.time())}.json",
                                       mime="application/json", use_container_width=True)
                except Exception as e:
                    st.error(f"JSON export failed: {e}")
        with c2:
            if st.button("📄 Markdown"):
                try:
                    md = exp.export_to_markdown(data)
                    st.download_button("Download MD", md, file_name=f"chat_{int(time.time())}.md",
                                       mime="text/markdown", use_container_width=True)
                except Exception as e:
                    st.error(f"Markdown export failed: {e}")

