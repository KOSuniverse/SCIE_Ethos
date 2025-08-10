# main.py — DROP-IN

import os
import io
import json
import uuid
import time
import ast
import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Make local modules importable (do this BEFORE importing project modules)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(str((Path(__file__).resolve().parent / "PY Files").resolve()))

# Orchestrator & session
from orchestrator import run_ingest_pipeline
from session import SessionState

# Optional import check for dev sanity
try:
    import column_alias
    import phase1_ingest.pipeline as _pcheck
    import importlib
    importlib.reload(column_alias)
    importlib.reload(_pcheck)
    st.sidebar.success("Imports OK: column_alias & phase1_ingest.pipeline")
except Exception as e:
    st.sidebar.error(f"Import check failed: {e}")

# Cloud helpers & paths
from path_utils import get_project_paths  # may be used for diagnostics/manifest
from dbx_utils import (
    list_xlsx as dbx_list_xlsx,
    read_file_bytes as dbx_read_bytes,
    save_xlsx_bytes,
    upload_bytes,
    upload_json,
)

# Pipeline adapter (cloud runner expects bytes, filename, paths)
from pipeline_adapter import run_pipeline_cloud as run_pipeline

# (Optional) Knowledge Base modules (unchanged)
from phase4_knowledge.knowledgebase_builder import status as kb_status, build_or_update_knowledgebase
from phase4_knowledge.knowledgebase_retriever import search_topk, pack_context
from phase4_knowledge.response_composer import compose_response

# --- Dropbox root auto-detect helper ---
def _detect_dropbox_root(list_func):
    """
    Try both Dropbox app modes and return the first root that works.
    Returns (root, raw_path, error_details | None)
    """
    candidates = [
        "/Project_Root",                     # App Folder access
        "/Apps/Ethos LLM/Project_Root",     # Full Dropbox access
    ]
    errors = []
    for root in candidates:
        raw_path = "/".join([root.rstrip("/"), "04_Data", "00_Raw_Files"])
        try:
            # Will raise if path doesn't exist or is inaccessible
            list_func(raw_path)
            return root, raw_path, None
        except Exception as e:
            errors.append((root, str(e)))
    return None, None, errors

# =============================================================================
# App setup
# =============================================================================
st.set_page_config(page_title="LLM Inventory Assistant", layout="wide")
st.title("📊 LLM Inventory + KB-Enhanced Assistant")

# Resolve canonical cloud paths (diagnostic/manifest usage)
cloud_paths = get_project_paths()  # keep separate from AppPaths below

# =============================================================================
# Unified AppPaths (Local + Dropbox)
# =============================================================================
class AppPaths:
    """
    Provide both local-style paths and Dropbox-style paths.
    Set one of project_root_local or project_root_dropbox.
    """
    def __init__(self, project_root_local: str | None = None, project_root_dropbox: str | None = None):
        self.project_root_local = project_root_local
        self.project_root_dropbox = project_root_dropbox

        # Common subpaths
        self._sub = {
            "raw": ("04_Data", "00_Raw_Files"),
            "cleansed": ("04_Data", "01_Cleansed_Files"),
            "eda": ("04_Data", "02_EDA_Charts"),
            "summaries": ("04_Data", "03_Summaries"),
            "metadata": ("04_Data", "04_Metadata"),
            "merged": ("04_Data", "05_Merged_Comparisons"),
        }

        # Local paths (None if not provided)
        if self.project_root_local:
            self.raw_folder = os.path.join(self.project_root_local, *self._sub["raw"])
            self.cleansed_folder = os.path.join(self.project_root_local, *self._sub["cleansed"])
            self.eda_folder = os.path.join(self.project_root_local, *self._sub["eda"])
            self.summaries_folder = os.path.join(self.project_root_local, *self._sub["summaries"])
            self.metadata_folder = os.path.join(self.project_root_local, *self._sub["metadata"])
            self.merged_folder = os.path.join(self.project_root_local, *self._sub["merged"])
            self.alias_json = os.path.join(self.metadata_folder, "global_column_aliases.json")
            self.master_metadata_path = os.path.join(self.metadata_folder, "master_metadata_index.json")
        else:
            self.raw_folder = None
            self.cleansed_folder = None
            self.eda_folder = None
            self.summaries_folder = None
            self.metadata_folder = None
            self.merged_folder = None
            self.alias_json = None
            self.master_metadata_path = None

        # Dropbox paths (None if not provided)
        if self.project_root_dropbox:
            def dbx(*parts): return "/".join((self.project_root_dropbox.rstrip("/"),) + parts)
            self.dbx_raw_folder = dbx(*self._sub["raw"])
            self.dbx_cleansed_folder = dbx(*self._sub["cleansed"])
            self.dbx_eda_folder = dbx(*self._sub["eda"])
            self.dbx_summaries_folder = dbx(*self._sub["summaries"])
            self.dbx_metadata_folder = dbx(*self._sub["metadata"])
            self.dbx_merged_folder = dbx(*self._sub["merged"])
            self.dbx_alias_json = dbx(*self._sub["metadata"], "global_column_aliases.json")
            self.dbx_master_metadata_path = dbx(*self._sub["metadata"], "master_metadata_index.json")
        else:
            self.dbx_raw_folder = None
            self.dbx_cleansed_folder = None
            self.dbx_eda_folder = None
            self.dbx_summaries_folder = None
            self.dbx_metadata_folder = None
            self.dbx_merged_folder = None
            self.dbx_alias_json = None
            self.dbx_master_metadata_path = None

# =============================================================================
# Ingest (debug) — local upload path
# =============================================================================
# =============================================================================
# Ingest (debug) — local upload path
# =============================================================================
with st.expander("🔧 Ingest pipeline (debug)"):
    root_mode = st.radio("Path mode", ["Local", "Dropbox"], horizontal=True)
    default_local = "/content/drive/MyDrive/Ethos LLM/Project_Root"
    default_dbx = "/Apps/Ethos LLM/Project_Root"

    # keep a stable place to store the chosen dropbox root
    if "dbx_root" not in st.session_state:
        st.session_state["dbx_root"] = default_dbx

    if root_mode == "Local":
        project_root_local = st.text_input("Project_Root (local)", value=default_local)
        app_paths = AppPaths(project_root_local=project_root_local)
    else:
        # Dropbox mode
        col1, col2 = st.columns([3,1])
        with col1:
            project_root_dropbox = st.text_input(
                "Project_Root (Dropbox)",
                value=st.session_state["dbx_root"]
            )
        with col2:
            if st.button("🔍 Auto-detect"):
                detected_root, detected_raw, errs = _detect_dropbox_root(dbx_list_xlsx)
                if detected_root:
                    st.session_state["dbx_root"] = detected_root
                    project_root_dropbox = detected_root
                    st.success(f"Detected Dropbox root: {detected_root}")
                    st.caption(f"RAW path: {detected_raw}")
                else:
                    st.error("Could not detect a valid Dropbox root.")
                    if errs:
                        with st.expander("Show detection errors"):
                            for r, msg in errs:
                                st.write(f"- {r}: {msg}")

        app_paths = AppPaths(project_root_dropbox=project_root_dropbox)

    # --- Local file upload path (unchanged) ---
    up = st.file_uploader("Upload an Excel file to ingest (local test)", type=["xlsx", "xlsm"])
    run_btn = st.button("Run Ingest Pipeline (local upload)")

    if run_btn and up is not None:
        file_bytes = up.read()
        cleaned_sheets, meta = run_ingest_pipeline(
            source=file_bytes,
            filename=up.name,
            paths=app_paths
        )

        st.subheader("Run metadata")
        st.json(meta)

        st.subheader("Sheets cleaned")
        st.write(list(cleaned_sheets.keys()))

        if "sheets" in meta:
            st.markdown("### Per-sheet summaries")
            rows = []
            for s in meta["sheets"]:
                rows.append({
                    "sheet_name": s.get("sheet_name"),
                    "normalized_sheet_type": s.get("normalized_sheet_type"),
                    "records": s.get("record_count"),
                    "summary": s.get("summary_text")
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


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
    st.caption("Switch the ingest expander to 'Dropbox' mode to enable listing.")
    st.code(getattr(app_paths, "dbx_raw_folder", None) or "(Dropbox root not set)", language="text")
    try:
        raw_dbx = getattr(app_paths, "dbx_raw_folder", None)
        if raw_dbx:
            raw_preview = dbx_list_xlsx(raw_dbx)
            st.caption(f"RAW .xlsx files found: {len(raw_preview)}")
        else:
            st.info("Dropbox mode not active.")
    except Exception as e:
        st.error(f"Dropbox list failed: {e}")

# =============================================================================
# RAW → Cleansed run (Dropbox ➜ Dropbox)
# =============================================================================
st.header("1) Process a RAW workbook (Dropbox ➜ Dropbox)")
raw_files = []
try:
    raw_dbx = getattr(app_paths, "dbx_raw_folder", None)
    if raw_dbx:
        raw_files = dbx_list_xlsx(raw_dbx)
    else:
        st.info("Switch to 'Dropbox' mode in the ingest expander to process cloud files.")
except Exception as e:
    st.error(f"Could not list RAW files: {e}")

if raw_files:
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

            # 1) Read bytes from Dropbox
            b = dbx_read_bytes(sel)

            st.info(f"Running pipeline for: {filename}")
            # 2) Run pipeline (bytes, filename, paths)
            cleaned_sheets, metadata = run_pipeline(b, filename, app_paths)

            # 3) Save cleansed workbook back to Dropbox
            out_bytes = save_xlsx_bytes(cleaned_sheets)
            out_base = filename.rsplit(".xlsx", 1)[0]
            cleansed_dbx = getattr(app_paths, "dbx_cleansed_folder", None)
            if not cleansed_dbx:
                raise RuntimeError("Dropbox cleansed folder not set (switch to 'Dropbox' mode).")
            out_path = f"{cleansed_dbx}/{out_base}_cleansed.xlsx"
            upload_bytes(out_path, out_bytes)
            st.success(f"✅ Cleansed workbook saved to: {out_path}")

            # 4) Update master metadata on Dropbox
            meta_path = getattr(app_paths, "dbx_master_metadata_path", None)
            if not meta_path:
                raise RuntimeError("Dropbox metadata path not set.")
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
else:
    st.info("No RAW files found (or not in Dropbox mode).")

# =============================================================================
# Browse Cleansed (Dropbox)
# =============================================================================
st.header("2) Browse Cleansed files (Dropbox)")
cln_files = []
try:
    cln_dbx = getattr(app_paths, "dbx_cleansed_folder", None)
    if cln_dbx:
        cln_files = dbx_list_xlsx(cln_dbx)
    else:
        st.info("Switch to 'Dropbox' mode to browse Cleansed files.")
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
    st.info("No files in Cleansed yet. Run step 1 first (Dropbox mode).")

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
st.text_input("Your question", value="", disabled=True)
st.button("Run Query", disabled=True)

# =============================================================================
# Project Manifest (diagnostic)
# =============================================================================
def scan_python_public_api(root: str):
    root_path = Path(root)
    manifest = {"scanned_at": time.time(), "root": str(root_path), "files": []}
    for p in root_path.rglob("*.py"):
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
            funcs = []
            classes = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    args = [a.arg for a in node.args.args]
                    funcs.append({"name": node.name, "args": args})
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    methods = []
                    for b in node.body:
                        if isinstance(b, ast.FunctionDef) and not b.name.startswith("_"):
                            margs = [a.arg for a in b.args.args]
                            methods.append({"name": b.name, "args": margs})
                    classes.append({"name": b.name, "methods": methods})
            manifest["files"].append({
                "path": str(p.relative_to(root_path)),
                "functions": funcs,
                "classes": classes
            })
        except Exception as e:
            manifest["files"].append({"path": str(p.relative_to(root_path)), "error": str(e)})
    return manifest

with st.expander("🧭 Project manifest (repo snapshot)"):
    code_root = str(Path(__file__).resolve().parent / "PY Files")
    st.caption(f"Scanning: {code_root}")
    if st.button("Generate manifest"):
        mf = scan_python_public_api(code_root)
        st.success("Manifest generated.")
        st.json(mf)

        # Save to Dropbox & S3 for shared truth
        try:
            dbx_manifest_path = f"{cloud_paths.metadata_folder}/project_manifest.json"
            upload_json(dbx_manifest_path, mf)
            st.caption(f"Saved to Dropbox: {dbx_manifest_path}")
        except Exception as e:
            st.warning(f"Dropbox save failed: {e}")

        try:
            import boto3
            s3 = boto3.client(
                "s3",
                region_name=st.secrets["AWS_DEFAULT_REGION"],
                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            )
            bucket = st.secrets["S3_BUCKET"]
            prefix = st.secrets["S3_PREFIX"].rstrip("/")
            key = f"{prefix}/04_Data/04_Metadata/project_manifest.json"
            s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(mf).encode("utf-8"))
            st.caption(f"Saved to S3: s3://{bucket}/{key}")
        except Exception as e:
            st.warning(f"S3 save failed: {e}")
