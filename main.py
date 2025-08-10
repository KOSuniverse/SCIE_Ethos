import os
import io
from pathlib import Path
import json
import uuid
import datetime
import pandas as pd
import streamlit as st

# Make local modules importable
from orchestrator import run_ingest_pipeline
from session import SessionState
import sys
from pathlib import Path
sys.path.append(str((Path(__file__).resolve().parent / "PY Files").resolve()))
try:
    import column_alias
    import phase1_ingest.pipeline as _pcheck
    import importlib; importlib.reload(column_alias); importlib.reload(_pcheck)
    st.sidebar.success("Imports OK: column_alias & phase1_ingest.pipeline")
except Exception as e:
    st.sidebar.error(f"Import check failed: {e}")


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
from pipeline_adapter import run_pipeline_cloud as run_pipeline

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

# If you already have a function for app Paths, reuse it.
class AppPaths:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.metadata_folder = os.path.join(project_root, "04_Data", "04_Metadata")
        self.alias_json = os.path.join(self.metadata_folder, "global_column_aliases.json")

# ---- Ingest (debug) UI block ----
with st.expander("🔧 Ingest pipeline (debug)"):
    project_root = st.text_input(
        "Project_Root",
        value="/content/drive/MyDrive/Ethos LLM/Project_Root"
    )
    paths = AppPaths(project_root)

    up = st.file_uploader("Upload an Excel file to ingest", type=["xlsx", "xlsm"])
    run_btn = st.button("Run Ingest Pipeline")

    if run_btn and up is not None:
        file_bytes = up.read()
        cleaned_sheets, meta = run_ingest_pipeline(
            source=file_bytes,
            filename=up.name,
            paths=paths
        )

        st.subheader("Run metadata")
        st.json(meta)

        st.subheader("Sheets cleaned")
        st.write(list(cleaned_sheets.keys()))

        # Per-sheet preview + summary
        for sname, df in cleaned_sheets.items():
            st.markdown(f"### Sheet: `{sname}`")
            st.write(df.head(10))

        # Optional: show quick summaries table
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
            st.dataframe(pd.DataFrame(rows))

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

# ===== Project Manifest (diagnostic) =====
import ast, os, json, time
from pathlib import Path
import streamlit as st

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
                    classes.append({"name": node.name, "methods": methods})
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
            from dbx_utils import upload_json
            from path_utils import get_project_paths
            paths = get_project_paths()
            dbx_manifest_path = f"{paths.metadata_folder}/project_manifest.json"
            upload_json(dbx_manifest_path, mf)
            st.caption(f"Saved to Dropbox: {dbx_manifest_path}")
        except Exception as e:
            st.warning(f"Dropbox save failed: {e}")

        try:
            import boto3, uuid
            s3 = boto3.client(
                "s3",
                region_name=st.secrets["AWS_DEFAULT_REGION"],
                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            )
            bucket = st.secrets["S3_BUCKET"]; prefix = st.secrets["S3_PREFIX"].rstrip("/")
            key = f"{prefix}/04_Data/04_Metadata/project_manifest.json"
            s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(mf).encode("utf-8"))
            st.caption(f"Saved to S3: s3://{bucket}/{key}")
        except Exception as e:
            st.warning(f"S3 save failed: {e}")
