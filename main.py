import streamlit as st
import io, pandas as pd 
from dbx_utils import list_xlsx as dbx_list_xlsx, read_file_bytes as dbx_read_bytes
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

# --- S3 health check (sidebar) ---
import boto3, uuid, datetime

with st.sidebar:
    st.subheader("S3 (health check)")
    if st.button("Run S3 check"):
        try:
            s3 = boto3.client(
                "s3",
                region_name=st.secrets["AWS_DEFAULT_REGION"],
                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            )
            bucket = st.secrets["S3_BUCKET"]
            prefix = st.secrets["S3_PREFIX"].rstrip("/")
            run_id = str(uuid.uuid4())[:8]
            key = f"{prefix}/_healthcheck/{st.secrets.get('ENV_NAME','prod')}/{run_id}.txt"

            # write + read
            body = f"ok {datetime.datetime.utcnow().isoformat()}Z".encode()
            s3.put_object(Bucket=bucket, Key=key, Body=body)
            got = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()

            st.success(f"S3 OK → wrote & read {key}")
            st.caption(got)
        except Exception as e:
            st.error(f"S3 check failed: {e}")

# --- Dropbox path sanity check (sidebar) ---
from typing import Optional
import dropbox

def _dbx_client() -> dropbox.Dropbox:
    return dropbox.Dropbox(
        oauth2_refresh_token=st.secrets["DROPBOX_REFRESH_TOKEN"],
        app_key=st.secrets["DROPBOX_APP_KEY"],
        app_secret=st.secrets["DROPBOX_APP_SECRET"],
    )

with st.sidebar:
    st.subheader("Dropbox (path check)")
    # Build the canonical Raw folder from your secrets
    raw_path = f"{st.secrets.get('DROPBOX_ROOT','/Project_Root')}/04_Data/00_Raw_Files"


    st.code(raw_path, language="text")

    if st.button("List RAW files"):
        try:
            dbx = _dbx_client()
            resp = dbx.files_list_folder(raw_path)
            xlsx = [e.name for e in resp.entries if isinstance(e, dropbox.files.FileMetadata) and e.name.lower().endswith(".xlsx")]
            if xlsx:
                st.success(f"Found {len(xlsx)} .xlsx files")
                for name in xlsx[:25]:
                    st.write("•", name)
            else:
                st.warning("No .xlsx files found in that folder.")
        except Exception as e:
            st.error(f"Dropbox check failed: {e}")

with st.sidebar:
    st.subheader("RAW file → preview")
    raw_path = f"{st.secrets.get('DROPBOX_ROOT','/Project_Root')}/04_Data/00_Raw_Files"
    raw_files = dbx_list_xlsx(raw_path)

    if not raw_files:
        st.info("No RAW files found. Move a source file into 00_Raw_Files to proceed.")
    else:
        labels = [f'{f["name"]}  ·  {f["path_lower"]}' for f in raw_files]
        choice = st.selectbox("Pick a RAW workbook", options=labels, index=0)

        if st.button("Load workbook"):
            sel = raw_files[labels.index(choice)]["path_lower"]
            b = dbx_read_bytes(sel)
            xls = pd.ExcelFile(io.BytesIO(b))
            st.success(f"Loaded: {sel}")
            st.write("Sheets:", xls.sheet_names)

            # optional: quick peek at first sheet
            try:
                first_sheet = xls.sheet_names[0]
                df_preview = pd.read_excel(io.BytesIO(b), sheet_name=first_sheet, nrows=5)
                st.caption(f"Preview: {first_sheet}")
                st.dataframe(df_preview)
            except Exception as e:
                st.warning(f"Preview failed: {e}")


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

