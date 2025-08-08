import streamlit as st
import pandas as pd
import os
import sys
sys.path.append("PY Files")

from llm_client import get_openai_client
from orchestrator import run_query_pipeline
from file_utils import list_cleaned_files, get_metadata_path, ensure_folder_structure
from loader import load_excel_file
from session import SessionState
from constants import SESSION_LOG_FILE
from pathlib import Path

# Phase-4 KB imports
from phase4_knowledge.knowledgebase_builder import status as kb_status, build_or_update_knowledgebase
from phase4_knowledge.knowledgebase_retriever import search_topk, pack_context
from phase4_knowledge.response_composer import compose_response

# --- INITIAL SETUP ---
PROJECT_ROOT = st.secrets.get("PROJECT_ROOT", str(Path(__file__).resolve().parent))
ensure_folder_structure(PROJECT_ROOT)

st.set_page_config(page_title="LLM Inventory Assistant", layout="wide")
st.title("📊 LLM Inventory + KB-Enhanced Assistant")

# --- SESSION STATE ---
if "session" not in st.session_state:
    st.session_state.session = SessionState()

session = st.session_state.session

# --- SIDEBAR: KB CONTROLS ---
st.sidebar.header("Knowledge Base")
kb_stat = kb_status(PROJECT_ROOT)
st.sidebar.json(kb_stat)

include_text = st.sidebar.checkbox("Include .txt/.md", value=True)
force_rebuild = st.sidebar.checkbox("Force full rebuild", value=False)
if st.sidebar.button("🔧 Build / Update KB"):
    with st.spinner("Building knowledge base..."):
        res = build_or_update_knowledgebase(
            project_root=PROJECT_ROOT,
            scan_folders=None,
            force_rebuild=force_rebuild,
            include_text_files=include_text
        )
    st.sidebar.success("KB build complete")
    st.sidebar.json(res)

# --- FILE SELECTION ---
cleaned_files = list_cleaned_files(PROJECT_ROOT)
file_selection = st.selectbox("📂 Choose a cleansed Excel file:", cleaned_files)

if file_selection:
    raw_obj = load_excel_file(file_selection)

    # --- Normalize to {sheet_name: DataFrame} ---
    def _normalize_excel(obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, pd.DataFrame):
            return {"Sheet1": obj}
        # Pandas ExcelFile or path-like
        try:
            if hasattr(obj, "sheet_names"):
                return {name: obj.parse(name) for name in obj.sheet_names}
        except Exception:
            pass
        try:
            xls = pd.ExcelFile(obj)  # obj might be a filepath
            return {name: xls.parse(name) for name in xls.sheet_names}
        except Exception:
            return {}

    df_dict = _normalize_excel(raw_obj)

    if not df_dict:
        st.error("❌ Could not read any sheets from the selected Excel file.")
    else:
        st.success(f"Loaded file: `{os.path.basename(file_selection)}`")
        st.write(f"Available sheets: {list(df_dict.keys())}")


    # Show sheets loaded
    st.success(f"Loaded file: `{os.path.basename(file_selection)}`")
    st.write(f"Available sheets: {list(df_dict.keys())}")

    # --- USER QUESTION ---
    user_query = st.text_area("🔍 Ask a question about this data:", height=100)

    if st.button("Run Query") and user_query:
        with st.spinner("Thinking..."):

            # 1. Run your existing EDA/data pipeline
            metadata_path = get_metadata_path(PROJECT_ROOT)
            eda_result = run_query_pipeline(user_query, df_dict, metadata_path)

            # 2. KB retrieval
            hits = search_topk(PROJECT_ROOT, user_query, k=5)
            kb_context = pack_context(hits, max_tokens=1000)

            # 3. Merge EDA reasoning + KB context
            combined_context = ""
            if "reasoning" in eda_result:
                combined_context += f"### Data Reasoning:\n{eda_result['reasoning']}\n\n"
            if kb_context.strip():
                combined_context += f"### Knowledge Base Context:\n{kb_context}"

            # 4. Get final LLM answer
            final_answer = compose_response(
                query=user_query,
                context=combined_context,
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=800
            )

            # Save to session
            session.add_entry(user_query, {
                "output": final_answer,
                "eda_result": eda_result,
                "kb_hits": hits
            })

        # --- OUTPUT ---
        st.subheader("💡 Final Answer")
        st.write(final_answer)

        # --- Expanders ---
        if "reasoning" in eda_result:
            with st.expander("🧠 Data Reasoning"):
                st.markdown(eda_result["reasoning"])

        with st.expander("📄 Matched File/Sheet"):
            st.json({
                "intent": eda_result.get("intent"),
                "file": eda_result.get("matched_file"),
                "sheet": eda_result.get("matched_sheet")
            })

        with st.expander("📚 KB Sources"):
            st.write(f"Retrieved {len(hits)} chunks")
            for h in hits:
                st.markdown(f"**Score:** {h.score:.3f} — `{h.meta.get('file_path', '')}`")
                st.write(h.text[:500] + ("..." if len(h.text) > 500 else ""))

        # Save session log
        session.save_log_to_file(os.path.join(PROJECT_ROOT, SESSION_LOG_FILE))

