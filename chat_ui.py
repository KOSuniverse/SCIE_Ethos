# chat_ui.py — Architecture-Compliant Streamlit Chat Interface (drop-in)
# Preserves your UI & layout; adds universal KB Summaries engine with drill-down ON by default.
# Does NOT modify DP mode or main.py.

import os
import json
import time
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Make local modules importable (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(str((Path(__file__).resolve().parent / "PY Files").resolve()))

# Core enterprise modules (unchanged)
from constants import PROJECT_ROOT, META_DIR, DATA_ROOT, CLEANSED
from session import SessionState
from logger import log_event, log_query_result
from orchestrator import answer_question
from assistant_bridge import auto_model, run_query, run_query_with_files
from confidence import score_ravc, should_abstain, score_confidence_enhanced, get_service_level_zscore, get_confidence_badge
from path_utils import get_project_paths
from dbx_utils import list_data_files
from export_utils import ExportManager
from sources_drawer import SourcesDrawer
from data_needed_panel import DataNeededPanel

# DO NOT call set_page_config here; main.py owns it. We only add theme-safe CSS.

# Existing CSS (unchanged)
st.markdown("""
<style>
    .stChatMessage {
        font-size: 1rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .badge {
        padding: 4px 8px;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .confidence-high { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }
    .confidence-medium { background: linear-gradient(135deg, #FF9800 0%, #f57c00 100%); }
    .confidence-low { background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%); }
    .model-mini { background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); }
    .model-4o { background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%); }
    .model-assistant { background: linear-gradient(135deg, #00BCD4 0%, #0097A7 100%); }

    .sources-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }

    .service-level-badge {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }

    .stButton button { border-radius: 8px; transition: all 0.3s ease; font-weight: 500; }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .stChatInput input { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THEME VAR SHIM — keep your layout but make dark mode actually honor theme
# ─────────────────────────────────────────────────────────────────────────────
_bg  = st.get_option("theme.backgroundColor") or "#0e1117"
_sb  = st.get_option("theme.secondaryBackgroundColor") or "#1b1f24"
_tx  = st.get_option("theme.textColor") or "#e6edf3"
_bd  = "rgba(255,255,255,0.12)"  # subtle border for dark

st.markdown(f"""
<style>
:root {{
  --secondary-background-color: {_sb};
  --border-color: {_bd};
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = "default"
if "session_handler" not in st.session_state:
    st.session_state.session_handler = SessionState()
if "last_sources" not in st.session_state:
    st.session_state.last_sources = {}
if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = []
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []
if "service_level" not in st.session_state:
    st.session_state.service_level = 0.95
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def get_service_level_badge(service_level: float) -> str:
    z_score = get_service_level_zscore(service_level)
    return f'<span class="service-level-badge">{service_level:.1%} (z={z_score:.3f})</span>'

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Configuration (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 SCIE Ethos Control Panel")
    st.info("💡 **Tip**: Toggle dark/light mode using the ⚙️ settings menu (top-right)")

    st.subheader("⚙️ Service Level")
    service_level = 0.95
    if service_level != st.session_state.service_level:
        st.session_state.service_level = service_level
        st.rerun()
    service_level_badge = get_service_level_badge(service_level)
    st.markdown(f"**Current:** {service_level_badge}", unsafe_allow_html=True)
    z_score = get_service_level_zscore(service_level)
    st.caption(f"Z-score: {z_score:.3f} (confidence interval)")
    st.markdown("---")

    # Upload area (unchanged)
    st.subheader("📎 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload documents for analysis",
        type=['pdf', 'docx', 'pptx', 'txt', 'md', 'csv', 'xlsx'],
        accept_multiple_files=True,
        help="Upload documents to get specific answers from your files"
    )
    if uploaded_files:
        st.session_state.selected_files = uploaded_files
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for file in uploaded_files:
            st.write(f"📄 {file.name}")
    else:
        st.session_state.selected_files = []
    st.markdown("---")

    st.subheader("🤖 Model Selection")
    model_choice = "Auto (Recommended)"

    if st.session_state.confidence_history:
        st.subheader("📊 Confidence History")
        st.line_chart(st.session_state.confidence_history)

    st.markdown("---")
    st.subheader("📊 Data Needed & Gaps")
    if "data_needed_panel" not in st.session_state:
        st.session_state.data_needed_panel = DataNeededPanel()
    data_panel = st.session_state.data_needed_panel
    data_panel.load_from_session()
    data_panel.render_panel(expanded=False)

    st.subheader("📚 Knowledge Base Tools")
    if st.button("🔄 Index New Documents", help="Process new documents and create searchable summaries"):
        with st.spinner("Indexing new documents."):
            try:
                sys.path.append('PY Files')
                from kb_indexer import KBIndexer
                kb_path = os.getenv('KB_DBX_PATH', '/Project_Root/06_LLM_Knowledge_Base')
                data_path = os.getenv('DATA_DBX_PATH', '/Project_Root/04_Data')
                indexer = KBIndexer(kb_path, data_path)
                result = indexer.process_new_files()
                st.success(f"✅ Indexing Complete!")
                st.info(f"""📊 **Results:**
- Processed: {result.get('processed', 0)} new files
- Failed: {result.get('failed', 0)} files  
- Skipped (unchanged): {result.get('skipped', 0)} files
- Skipped (in FAISS): {result.get('faiss_skipped', 0)} files
- Total index size: {result.get('index_size', 0)} documents""")
            except Exception as e:
                st.error(f"❌ Indexing failed: {str(e)}")
    st.caption("Creates searchable summaries for new documents while respecting your existing FAISS index.")

# ─────────────────────────────────────────────────────────────────────────────
# === KB SUMMARIES ENGINE (universal; drill-down default ON; no special-cases) ===
# ─────────────────────────────────────────────────────────────────────────────
import math
from collections import Counter

# Optional deep-read libs (already imported; we keep graceful fallbacks)
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

def _tok(s: str):
    return [w.lower() for w in re.findall(r"[A-Za-z0-9]+", s or "")]

def _bm25(q, d, avgdl, k1=1.5, b=0.75):
    tf = Counter(d); dl = len(d); sc = 0.0
    for t, idf in q:
        f = tf.get(t, 0)
        if f:
            sc += idf * ((f*(k1+1)) / (f + k1*(1-b+b*dl/avgdl)))
    return sc

def _load_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _collect_text(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_text(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_text(v, out)

KB_ROOT_DIR      = Path(os.getenv("PROJECT_ROOT", "/Project_Root")) / "06_LLM_Knowledge_Base"
KB_SUMMARIES_DIR = KB_ROOT_DIR / "KB_Summaries"
KB_MEETING_DIR   = KB_ROOT_DIR / "Meeting Minutes"
KB_EMAIL_DIR     = KB_ROOT_DIR / "Email"

@st.cache_resource(show_spinner=False)
def _build_kb_index():
    docs=[]
    if KB_SUMMARIES_DIR.exists():
        for p in KB_SUMMARIES_DIR.rglob("*_summary.json"):
            js = _load_json(p)
            if not isinstance(js, dict): continue
            buf=[]; _collect_text(js, buf)
            docs.append({"path":p, "file_name": js.get("file_name") or p.name,
                         "doc_type": js.get("document_type",""), "js": js,
                         "text":"\n".join(buf)})
    # Tokenize/IDF
    toks=[]; df=Counter()
    for d in docs:
        td=_tok(d["text"]); toks.append(td); df.update(set(td))
    N=len(docs); avgdl=(sum(len(t) for t in toks)/N) if N else 1
    idf={t: math.log((N-df[t]+0.5)/(df[t]+0.5)+1) for t in df}
    return {"docs":docs, "toks":toks, "idf":idf, "avgdl":avgdl, "N":N}

def _search_kb(index, query, topk=6):
    if index["N"]==0: return []
    qt=_tok(query); q=[(t, index["idf"].get(t,0.0)) for t in qt]
    scored=[]
    for i, dt in enumerate(index["toks"]):
        scored.append((_bm25(q, dt, index["avgdl"]), i))
    scored.sort(reverse=True)
    return [(index["docs"][i], s) for s,i in scored[:topk] if s>0]

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
    # Priority: explicit sources relative to KB root
    texts=[]
    for k in ("sources","source_files"):
        arr=js.get(k,[])
        if isinstance(arr, list):
            for rel in arr:
                pp=(KB_ROOT_DIR/rel) if not Path(rel).is_absolute() else Path(rel)
                suf=pp.suffix.lower()
                if suf==".pdf": texts.append(_deep_pdf(pp))
                elif suf==".docx": texts.append(_deep_docx(pp))
                elif suf==".msg": texts.append(_deep_msg(pp))
    if texts: return "\n".join(t for t in texts if t)

    # Fallback: stem-match file_name in Meeting Minutes / Email / KB root
    stem=Path(js.get("file_name","")).stem
    for folder in (KB_MEETING_DIR, KB_EMAIL_DIR, KB_ROOT_DIR):
        for gp in (list(folder.rglob(f"{stem}*.pdf"))[:1] +
                   list(folder.rglob(f"{stem}*.docx"))[:1] +
                   list(folder.rglob(f"{stem}*.msg"))[:1]):
            suf=gp.suffix.lower()
            if suf==".pdf": return _deep_pdf(gp)
            if suf==".docx": return _deep_docx(gp)
            if suf==".msg": return _deep_msg(gp)
    return ""

def kb_summaries_answer(prompt: str, deep_read=True) -> Optional[str]:
    idx=_build_kb_index()
    hits=_search_kb(idx, prompt, topk=8)
    if not hits:
        return None

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
            act=[]
            for a in js["actions"][:10]:
                if isinstance(a, dict):
                    who=a.get("owner"); item=a.get("item") or a.get("action") or ""
                    due=a.get("due")
                    line=f"- {item}"
                    if who: line=f"- ({who}) {item}"
                    if due: line += f" — due {due}"
                    act.append(line)
                elif isinstance(a,str):
                    act.append("- " + a)
            if act: parts.append("**Actions:**\n" + "\n".join(act))

        if js.get("summary"):
            parts.append("**Summary:** " + js["summary"])

        if deep_read:
            extra=_drill_sources(js)
            if extra:
                parts.append("**Source Excerpt:** " + extra[:1200])

        if parts:
            blocks.append(
                f"<div class='sources-card'>"
                f"<div class='badge'>{js.get('document_type','Summary')}</div> "
                f"<div class='badge'>{d['file_name']}</div><br/>"
                + "<br/>".join(parts) +
                f"</div>"
            )

    return ("**📚 Knowledge Base Answer**\n\n" + "\n\n".join(blocks)) if blocks else None

# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Interface (unchanged layout)
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧠 SCIE Ethos LLM Assistant")
st.caption("Architecture-Compliant Chat Interface with Enhanced Phase 4 Features")

# Confidence badge render (unchanged)
if st.session_state.confidence_history:
    try:
        last_confidence = st.session_state.confidence_history[-1]
        if isinstance(last_confidence, dict):
            confidence_value = last_confidence.get("score", 0.5)
        else:
            confidence_value = float(last_confidence) if last_confidence is not None else 0.5
        confidence_badge_html = get_confidence_badge(confidence_value, st.session_state.service_level)
        st.markdown(f"**Confidence:** {confidence_badge_html}", unsafe_allow_html=True)
    except Exception:
        st.markdown("**Confidence:** Medium (0.50)", unsafe_allow_html=True)

# History display (unchanged)
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("artifacts"):
            st.markdown("**📎 Generated Artifacts:**")
            for artifact in message["artifacts"]:
                st.code(str(artifact))

# Chat input (pinned at bottom — unchanged)
if prompt := st.chat_input("Ask about inventory, WIP, E&O, forecasting, or root cause analysis."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            export_manager = ExportManager(st.session_state.service_level)

            # Your existing model flow
            if model_choice == "Assistant API":
                response = run_query(prompt, selected_files=st.session_state.selected_files)
            else:
                model_name = "gpt-4o" if model_choice == "GPT-4o" else "gpt-5"
                response = run_query(prompt, model=model_name, selected_files=st.session_state.selected_files)

            answer = response.get("answer", "")
            sources = response.get("sources", {})
            confidence_score = response.get("confidence", 0.5)

            # Enterprise KB Summaries: drill-down ON by default
            kb_ans = kb_summaries_answer(prompt, deep_read=True)

            # Quality-first behavior:
            # If model gave nothing / "no relevant" -> show KB summaries answer.
            if (not answer or answer.strip() == "" or "no relevant" in answer.strip().lower()):
                if kb_ans:
                    answer = kb_ans

            # (Optional) If you want to ALWAYS show the KB answer above the model answer, replace the if-block above with:
            # if kb_ans:
            #     st.markdown(kb_ans, unsafe_allow_html=True)
            # Then also render the model answer below.

            # Save confidence
            st.session_state.confidence_history.append(confidence_score)
            if len(st.session_state.confidence_history) > 10:
                st.session_state.confidence_history.pop(0)

            st.session_state.last_sources = sources

            # Render either templated sections or plain markdown
            template_sections = response.get("template", {})
            if template_sections and len(template_sections) > 2 and not (answer or "").startswith("**📚 Knowledge Base Answer**"):
                st.markdown("## " + template_sections.get("title", "Analysis Results"))
                if template_sections.get("executive_insight"):
                    st.markdown("### Executive Insight")
                    st.markdown(template_sections["executive_insight"])
                if template_sections.get("analysis"):
                    st.markdown("### Detailed Analysis")
                    st.markdown(template_sections["analysis"])
                if template_sections.get("recommendations"):
                    st.markdown("### Recommendations")
                    st.markdown(template_sections["recommendations"])
                if template_sections.get("citations"):
                    st.markdown("### Citations")
                    st.markdown(template_sections["citations"])
                if template_sections.get("limits_data_needed"):
                    st.markdown("### Limits/Missing Data")
                    st.markdown(template_sections["limits_data_needed"])
            else:
                message_placeholder.markdown(answer if answer else "No answer provided", unsafe_allow_html=True)

            # Sources drawer (unchanged)
            sources_drawer = SourcesDrawer()
            coverage_warning = response.get("kb_coverage_warning")
            if coverage_warning:
                st.warning(f"⚠️ {coverage_warning}")
            sources_drawer.render_inline_sources(sources, confidence_score)

            # Save assistant message
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer if answer else "No answer provided",
                "sources": sources,
                "confidence": confidence_score,
                "service_level": st.session_state.service_level
            })

        except Exception as e:
            msg = f"❌ Error processing query: {str(e)}"
            message_placeholder.error(msg)
            st.session_state.chat_messages.append({"role": "assistant", "content": msg, "error": True})

# ─────────────────────────────────────────────────────────────────────────────
# Export Options (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chat_messages:
    st.markdown("---")
    st.header("📤 Export Options")
    export_manager = ExportManager(st.session_state.service_level)
    export_data = {
        "messages": st.session_state.chat_messages,
        "sources": st.session_state.last_sources,
        "confidence_history": st.session_state.confidence_history,
        "selected_files": st.session_state.selected_files,
        "metadata": {
            "conversation_id": st.session_state.current_conversation,
            "service_level": st.session_state.service_level,
            "exported_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_messages": len(st.session_state.chat_messages)
        }
    }
    if "data_needed_panel" in st.session_state:
        gaps_summary = st.session_state.data_needed_panel.get_gaps_summary()
        export_data["data_gaps"] = gaps_summary

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("📊 XLSX", use_container_width=True):
            try:
                xlsx_content = export_manager.export_to_xlsx(export_data)
                st.download_button("Download XLSX", xlsx_content,
                    file_name=f"scie_ethos_export_{int(time.time())}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
            except Exception as e:
                st.error(f"XLSX export failed: {e}")
    with col2:
        if st.button("📄 Markdown", use_container_width=True):
            try:
                md_content = export_manager.export_to_markdown(export_data)
                st.download_button("Download MD", md_content,
                    file_name=f"scie_ethos_export_{int(time.time())}.md",
                    mime="text/markdown",
                    use_container_width=True)
            except Exception as e:
                st.error(f"Markdown export failed: {e}")
    with col3:
        if st.button("📝 DOCX", use_container_width=True):
            try:
                docx_content = export_manager.export_to_docx(export_data)
                st.download_button("Download DOCX", docx_content,
                    file_name=f"scie_ethos_export_{int(time.time())}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True)
            except Exception as e:
                st.error(f"DOCX export failed: {e}")
    with col4:
        if st.button("📄 PDF", use_container_width=True):
            try:
                pdf_content = export_manager.export_to_pdf(export_data)
                st.download_button("Download PDF", pdf_content,
                    file_name=f"scie_ethos_export_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True)
            except Exception as e:
                st.error(f"PDF export failed: {e}")
    with col5:
        if st.button("🧾 JSON", use_container_width=True):
            try:
                json_content = export_manager.export_to_json(export_data)
                st.download_button("Download JSON", json_content,
                    file_name=f"scie_ethos_export_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True)
            except Exception as e:
                st.error(f"JSON export failed: {e}")
