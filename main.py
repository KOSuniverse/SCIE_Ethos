# main.py

import streamlit as st
import pandas as pd
import os

from llm_client import get_openai_client
from orchestrator import run_query_pipeline
from file_utils import list_cleaned_files, get_metadata_path, ensure_folder_structure
from loader import load_excel_file
from session import SessionState
from constants import SESSION_LOG_FILE

# --- INITIAL SETUP ---
PROJECT_ROOT = "Project_Root"
ensure_folder_structure(PROJECT_ROOT)

st.set_page_config(page_title="LLM Inventory Assistant", layout="wide")
st.title("📊 LLM Inventory & RCA Assistant")

# --- SESSION STATE ---
if "session" not in st.session_state:
    st.session_state.session = SessionState()

session = st.session_state.session

# --- FILE SELECTION ---
cleaned_files = list_cleaned_files(PROJECT_ROOT)
file_selection = st.selectbox("📂 Choose a cleansed Excel file:", cleaned_files)

if file_selection:
    df_dict = load_excel_file(file_selection)

    # Show sheets loaded
    st.success(f"Loaded file: `{os.path.basename(file_selection)}`")
    st.write(f"Available sheets: {list(df_dict.keys())}")

    # --- USER QUESTION ---
    user_query = st.text_area("🔍 Ask a question about this data:", height=100)

    if st.button("Run Query") and user_query:
        with st.spinner("Thinking..."):
            metadata_path = get_metadata_path(PROJECT_ROOT)
            result = run_query_pipeline(user_query, df_dict, metadata_path)

            session.add_entry(user_query, result)

        # --- OUTPUT ---
        st.subheader("💡 Answer")
        st.write(result.get("output", {}))

        if "reasoning" in result:
            with st.expander("🧠 Reasoning"):
                st.markdown(result["reasoning"])

        with st.expander("📄 Matched Context"):
            st.json({
                "intent": result.get("intent"),
                "file": result.get("matched_file"),
                "sheet": result.get("matched_sheet")
            })

        # Save session log
        session.save_log_to_file(os.path.join(PROJECT_ROOT, SESSION_LOG_FILE))
