import os
import io
from pathlib import Path
import json
import uuid
import datetime
import pandas as pd
import streamlit as st

# Make local modules importable
import sys
sys.path.append("PY Files")

# --- Cloud helpers & paths ---
from path_utils import get_project_paths
from dbx_utils import (
    list_xlsx as dbx_list_xlsx,
    read_file_bytes as dbx_read_bytes,
    save_xlsx_bytes,
    upload_bytes,
    upload_json,
)

# --- Optional: your pipeline entrypoint (uses your 30-file logic) ---
# TODO: if your function name or module differs, adjust this import line only.
from pipeline import run_pipeline

# --- (Optional) Knowledge Base modules (unchanged) ---
from phase4_knowledge.knowledgebase_builder import status as kb_status, build_or_update_knowledgebase
from phase4_knowledge.knowledgebase_retriever import search_topk, pack_context
from phase4_knowledge.response_composer import compose_response

# =============================================================================
# App setup
# =============================================================================
st.set_page_config(page_title="LLM Inventory Assistant", layout="wide")
st.title("📊 LLM Inventory + KB-Enhanced Assistant")

# Resolve canonical cloud paths
paths = get_project_paths()

# =============================================================================
# Sidebar: Cloud health checks
# =============================================================================
with st.sidebar:
    st.header("Cloud Health")

    # --- S3 health check ---
    import boto3
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
            body = f"ok {datetime.datetime.utcnow().isoformat()}Z".encode()
            s3.put_object(Bucket=bucket, Key=key, Body=body)
            got = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
            st.success(f"S3 OK → wrote & read {key}")
            st.caption(got)
        except Exception as e:
            st.error(f"S3 check failed: {e}")

    # --- Dropbox RAW listing (sanity) ---
    st.subheader("Dropbox (RAW path)")
    st.code(f"{paths.raw_folder}", language="text")
    try:
        raw_preview = dbx_list_xlsx(paths.raw_folder)
        st.caption(f"RAW .xlsx files found: {len(raw_preview)}")
    except Exception as e:
        st.error(f"Dropbox list failed: {e}")

# =============================================================================
# RAW → Cleansed run (cloud-only)
# =============================================================================
st.header("1) Process a RAW workbook (Dropbox ➜ Dropbox)")
raw_files = []
try:
    raw_files = dbx_list_xlsx(paths.raw_folder)
except Exception as e:
    st.error(f"Could not list RAW files: {e}")

if not raw_files:
    st.info("No RAW files found. Add a source file to Dropbox 00_Raw_Files and refresh.")
else:
    labels = [f'{f["name"]}  ·  {f["path_lower"]}' for f in raw_files]
    choice = st.selectbox("Pick a RAW workbook", options=labels, index=0)

    # Quick peek
    if st.button("🔎 Preview RAW workbook"):
        try:
            sel = raw_files[labels.index(choice)]["path_lower"]
            b = dbx_read_bytes(sel)
            xls = pd.ExcelFile(io.BytesIO(b))
            st.success(f"Loaded: {sel}")
            st.write("Sheets:", xls.sheet_names)
            # Peek first sheet
            if xls.sheet_names:
                first_sheet = xls.sheet_names[0]
                df_preview = pd.read_excel(io.BytesIO(b), sheet_name=first_sheet, nrows=5)
                st.caption(f"Preview: {first_sheet}")
                st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Preview failed: {e}")

    if st.button("🧹 Process & Save to Cleansed"):
        try:
            sel = raw_files[labels.index(choice)]["path_lower"]
            filename = raw_files[labels.index(choice)]["name"]

            st.info(f"Running pipeline for: {filename}")
            # IMPORTANT: run_pipeline should implement your original logic and return:
            # cleaned_sheets (dict[str, pd.DataFrame]), metadata (dict)
            cleaned_sheets, metadata = run_pipeline(filename, paths)

            # Save cleansed workbook (single multi-sheet file; your pipeline can also write per-type files inside)
            out_bytes = save_xlsx_bytes(cleaned_sheets)
            out_base = filename.rsplit(".xlsx", 1)[0]
            out_path = f"{paths.cleansed_folder}/{out_base}_cleansed.xlsx"
            upload_bytes(out_path, out_bytes)
            st.success(f"✅ Cleansed workbook saved to: {out_path}")

            # Update master_metadata_index.json
            meta_path = paths.master_metadata_path
            try:
                existing = json.loads(dbx_read_bytes(meta_path).decode("utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
            existing.append(metadata)
            upload_json(meta_path, existing)
            st.success(f"✅ Metadata updated at: {meta_path}")

            with st.expander("Sheets cleaned"):
                st.write(list(cleaned_sheets.keys()))
            with st.expander("Run metadata"):
                st.json(metadata)
        except Exception as e:
            st.error(f"Pipeline error: {e}")

# =============================================================================
# Browse Cleansed (Dropbox)
# =============================================================================
st.header("2) Browse Cleansed files (Dropbox)")
cln_files = []
try:
    cln_files = dbx_list_xlsx(paths.cleansed_folder)
except Exception as e:
    st.error(f"Could not list Cleansed files: {e}")

if cln_files:
    cln_labels = [f'{f["name"]}  ·  {f["path_lower"]}' for f in cln_files]
    cln_choice = st.selectbox("Pick a Cleansed workbook", options=cln_labels, index=0, key="cln_pick")
    if st.button("🔎 Preview Cleansed workbook"):
        try:
            sel = cln_files[cln_labels.index(cln_choice)]["path_lower"]
            b = dbx_read_bytes(sel)
            xls = pd.ExcelFile(io.BytesIO(b))
            st.success(f"Loaded: {sel}")
            st.write("Sheets:", xls.sheet_names)
            if xls.sheet_names:
                first_sheet = xls.sheet_names[0]
                df_preview = pd.read_excel(io.BytesIO(b), sheet_name=first_sheet, nrows=5)
                st.caption(f"Preview: {first_sheet}")
                st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Preview failed: {e}")
else:
    st.info("No files in Cleansed yet. Run step 1 first.")

# =============================================================================
# Knowledge Base (optional, unchanged)
# =============================================================================
st.header("3) Knowledge Base (optional)")
PROJECT_ROOT = st.secrets.get("PROJECT_ROOT", str(Path(__file__).resolve().parent))
with st.expander("Knowledge Base controls"):
    st.json(kb_status(PROJECT_ROOT))
    include_text = st.checkbox("Include .txt/.md", value=True)
    force_rebuild = st.checkbox("Force full rebuild", value=False)
    if st.button("🔧 Build / Update KB"):
        with st.spinner("Building knowledge base..."):
            res = build_or_update_knowledgebase(
                project_root=PROJECT_ROOT,
                scan_folders=None,
                force_rebuild=force_rebuild,
                include_text_files=include_text
            )
        st.success("KB build complete")
        st.json(res)

# =============================================================================
# Q&A over Cleansed (coming next)
# =============================================================================
st.header("4) Ask questions about a Cleansed workbook")
st.caption("This will be enabled after we switch the orchestrator/metadata loader to Dropbox paths.")

# --- Placeholder UI only (disabled until orchestrator is cloud-aware) ---
st.text_input("Your question", value="", disabled=True)
st.button("Run Query", disabled=True)

