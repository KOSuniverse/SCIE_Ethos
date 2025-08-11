# streamlit app for building and querying a knowledge base

import os, json, sys, streamlit as st
from pathlib import Path
sys.path.append("PY Files")  # ensure PY Files/* is importable
from phase4_knowledge.knowledgebase_builder import status, build_or_update_knowledgebase
from phase4_knowledge.kb_qa import kb_answer

# Dropbox-first default (app-root relative). Falls back to /Project_Root if unset.
_dropbox_root = st.secrets.get("DROPBOX_ROOT", os.getenv("DROPBOX_ROOT", "")).strip("/")
PROJECT_ROOT = f"/{_dropbox_root}" if _dropbox_root else "/Project_Root"

st.set_page_config(page_title="Ethos KB QA", layout="wide")
st.title("📚 Ethos Knowledge Base — QA")

# Key sanity
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.warning("Add OPENAI_API_KEY to Streamlit secrets.")
else:
    os.environ["OPENAI_API_KEY"] = api_key

with st.sidebar:
    st.header("Build / Update")
    include_text = st.checkbox("Include .txt/.md", value=True)
    force_rebuild = st.checkbox("Force full rebuild", value=False)
    if st.button("🔧 Build / Update KB", use_container_width=True):
        with st.spinner("Building…"):
            res = build_or_update_knowledgebase(
                project_root=PROJECT_ROOT,
                scan_folders=None,
                force_rebuild=force_rebuild,
                include_text_files=include_text
            )
        st.success("Build complete.")
        st.json(res)

    st.divider()
    st.header("Status")
    st.json(status(PROJECT_ROOT))  # uses updated relative paths inside Phase 4

st.subheader("Ask the Knowledge Base")
q = st.text_area("Question", placeholder="Ask anything contained in the uploaded docs…", height=100)
col1, col2, col3 = st.columns(3)
with col1:
    k = st.number_input("Top‑K", min_value=1, max_value=20, value=5, step=1)
with col2:
    max_ctx = st.number_input("Max context tokens", min_value=400, max_value=3000, value=1500, step=100)
with col3:
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

if st.button("💬 Get Answer", type="primary", use_container_width=True):
    if not q.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Thinking…"):
            res = kb_answer(
                project_root=PROJECT_ROOT,
                query=q.strip(),
                k=int(k),
                max_context_tokens=int(max_ctx),
                temperature=float(temp)
            )
        st.markdown("### ✅ Answer")
        st.write(res["answer"])

        with st.expander("Show context & sources"):
            st.markdown("**Packed context (truncated view):**")
            st.code(res["context"][:2000])
            st.markdown("**Top hits:**")
            hits_table = [
                {
                    "score": round(h.score, 3),
                    "path": h.meta.get("file_path", ""),
                    "type": h.meta.get("source_type", ""),
                    "pages": h.meta.get("page_range", None),
                    "tokens": h.meta.get("tokens_est", 0),
                } for h in res["hits"]
            ]
            st.dataframe(hits_table, use_container_width=True)
