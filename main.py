# main.py — DROP-IN

import os, io, json, uuid, time, ast, datetime, requests, sys
from pathlib import Path
import pandas as pd
import streamlit as st
# ─────────────────────────────────────────────────────────────────────────────
# Make local modules importable (do this BEFORE importing project modules)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(str((Path(__file__).resolve().parent / "PY Files").resolve()))

from phase1_ingest.pipeline import PIPELINE_VERSION
st.caption(f"Pipeline: {PIPELINE_VERSION}")

# Orchestrator & session
from orchestrator import ingest_streamlit_bytes_5  # add at top of file or above this block
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
    list_data_files as dbx_list_data_files,
    read_file_bytes as dbx_read_bytes,
    save_xlsx_bytes,
    upload_bytes,
    upload_json,
)

# Pipeline adapter (cloud runner expects bytes, filename, paths)
from pipeline_adapter import run_pipeline_cloud as run_pipeline

# (Optional) Knowledge Base modules (updated for cloud compatibility)
from phase4_knowledge.knowledgebase_builder import status as kb_status, build_or_update_knowledgebase, PROJECT_ROOT as KB_PROJECT_ROOT
from phase4_knowledge.knowledgebase_retriever import search_topk, pack_context
from phase4_knowledge.response_composer import compose_response

# Sources drawer and confidence badge for Data Processing mode
from sources_drawer import SourcesDrawer
from confidence import get_confidence_badge

# Phase 6: Production Infrastructure (Monitoring & Caching)
try:
    from monitoring_client import MonitoringClient
    from infrastructure.caching import CacheManager
    PHASE6_AVAILABLE = True
except ImportError as e:
    st.sidebar.warning(f"Phase 6 components not available: {e}")
    PHASE6_AVAILABLE = False

def build_xlsx_bytes_from_sheets(sheets: dict[str, pd.DataFrame]) -> bytes:
    import io
    import pandas as pd
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            safe_name = (str(sheet_name) or "Sheet1")[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    buf.seek(0)
    return buf.read()

from sidecars import backend_info
st.caption(f"Sidecars backend: {backend_info()}")

# Initialize Phase 6 components if available
if PHASE6_AVAILABLE:
    try:
        # Initialize monitoring client
        monitoring = MonitoringClient()
        st.sidebar.success("Phase 6: Monitoring enabled")
        
        # Initialize cache manager
        cache = CacheManager()
        st.sidebar.success("Phase 6: Caching enabled")
        
        # Monitor application startup
        monitoring.increment("app.starts", 1, environment="streamlit")
        monitoring.gauge("app.memory_usage", 0.0, environment="streamlit")  # Will be updated later
        
    except Exception as e:
        st.sidebar.error(f"Phase 6 initialization failed: {e}")
        PHASE6_AVAILABLE = False
else:
    # Fallback monitoring and caching
    monitoring = None
    cache = None

# --- Build a human-readable Markdown summary from pipeline metadata ---
def _build_summary_markdown(metadata: dict) -> str:
    import datetime as _dt
    fn = metadata.get("source_filename", "(unknown)")
    sheet_count = metadata.get("sheet_count", 0)
    exec_txt = metadata.get("executive_summary", "")
    ts = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%MZ")

    lines = [f"# Executive Summary — {fn}", "", f"_Generated: {ts}_", "", f"**Sheets:** {sheet_count}", ""]
    if exec_txt:
        lines += ["## Overview", exec_txt, ""]

    lines.append("## Per‑Sheet EDA")
    for s in metadata.get("sheets", []):
        lines.append(f"### {s.get('sheet_name')}")
        lines.append(f"- Type: `{s.get('normalized_sheet_type')}`")
        lines.append(f"- Records: {s.get('rows')}")
        
        # Only include EDA text if it's actual analysis, not raw metadata
        eda = (s.get("eda_text") or "").strip()
        if eda and not eda.startswith('{') and len(eda) > 50:  # Filter out raw JSON and short text
            lines.append("")
            lines.append(eda)
            lines.append("")
        else:
            # If no meaningful EDA text, show basic summary
            summary = s.get("summary_text", "").strip()
            if summary:
                lines.append("")
                lines.append(summary)
                lines.append("")
    return "\n".join(lines)


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
st.set_page_config(page_title="SCIE Ethos LLM", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Interface Mode Selection
# ─────────────────────────────────────────────────────────────────────────────
interface_mode = st.sidebar.radio(
    "🎛️ Interface Mode",
    ["💬 Chat Assistant", "🔧 Data Processing"],
    index=0,
    help="Choose between chat interface (architecture-compliant) or data processing workflows"
)

if interface_mode == "💬 Chat Assistant":
    # Auto-launch chat interface in new tab
    st.markdown("""
    ## 🧠 SCIE Ethos Chat Assistant
    
    **Architecture-Compliant Interface**
    - Assistants API as brain
    - File Search integration  
    - Confidence badges & model routing
    - Citation tracking & export capabilities
    
    Click the button below to automatically launch the chat interface in a new browser tab.
    """)
    
    if st.button("🚀 Launch Chat Assistant", type="primary", use_container_width=True):
        # Use subprocess to start chat_ui.py on a different port
        import subprocess
        import webbrowser
        import time
        import socket
        import requests
        from pathlib import Path
        
        def find_free_port():
            """Find a free port for the chat interface"""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        
        def wait_for_server(url, timeout=30):
            """Wait for the server to start responding"""
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        return True
                except:
                    pass
                time.sleep(1)
            return False
        
        try:
            # Find free port
            chat_port = find_free_port()
            
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            chat_ui_path = script_dir / "chat_ui.py"
            
            if not chat_ui_path.exists():
                st.error("chat_ui.py not found. Please ensure it's in the same directory as main.py")
                st.stop()
            
            with st.spinner(f"Starting chat interface on port {chat_port}..."):
                # Start chat interface on different port
                process = subprocess.Popen([
                    "streamlit", "run", str(chat_ui_path),
                    "--server.port", str(chat_port),
                    "--server.headless", "true",
                    "--server.enableCORS", "false"
                ], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Build URL
                chat_url = f"http://localhost:{chat_port}"
                
                # Wait for server to be ready
                st.info(f"Waiting for server to start at {chat_url}...")
                if wait_for_server(chat_url, timeout=30):
                    # Server is ready, open browser
                    webbrowser.open(chat_url)
                    st.success(f"✅ Chat Assistant launched successfully!")
                    st.info(f"🌐 Interface URL: {chat_url}")
                    st.info("The chat interface should open in a new browser tab automatically.")
                else:
                    # Server failed to start
                    process.terminate()
                    st.error("❌ Failed to start chat interface server")
                    
                    # Try to get error output
                    try:
                        stdout, stderr = process.communicate(timeout=5)
                        if stderr:
                            st.error(f"Error output: {stderr.decode()}")
                    except:
                        pass
                    
                    st.markdown(f"""
                    **Manual Launch Options:**
                    
                    1. **Option 1 - Try different port:**
                    ```bash
                    streamlit run chat_ui.py --server.port 8502
                    ```
                    Then visit: http://localhost:8502
                    
                    2. **Option 2 - Default port:**
                    ```bash
                    streamlit run chat_ui.py
                    ```
                    Then visit: http://localhost:8501
                    
                    3. **Option 3 - Check if port {chat_port} is available:**
                    ```bash
                    netstat -an | findstr {chat_port}
                    ```
                    """)
            
        except Exception as e:
            st.error(f"Failed to launch chat interface: {e}")
            st.markdown("""
            **Fallback: Manual Launch**
            
            Open a new terminal/command prompt and run:
            ```bash
            streamlit run chat_ui.py
            ```
            Then visit: http://localhost:8501
            """)
        
        # Show preview of chat features
        with st.expander("� Chat Interface Features"):
            st.markdown("""
            **Core Features:**
            - 🧠 **Assistants API Integration**: Full OpenAI Assistant with File Search
            - 📊 **File Selection**: Choose specific cleansed files for analysis  
            - 🎯 **Intent Classification**: Auto-routing to appropriate models
            - 📈 **Confidence Scoring**: R/A/V/C methodology with abstention
            - 💬 **Conversation Management**: Named conversations with history
            - 📎 **Artifact Handling**: Charts, data files, and analysis results
            - 📚 **Knowledge Base**: Integrated KB search and citations
            - 📄 **Export Options**: Markdown and JSON conversation exports
            
            **Architecture Compliance:**
            - ✅ Assistants API as primary brain
            - ✅ Dropbox → Assistant File Store sync
            - ✅ Cloud-first file handling
            - ✅ Confidence & abstention policies
            - ✅ Model auto-upgrade (mini → 4o) 
            - ✅ Citation tracking & sources
            """)
    
    st.stop()  # Don't show data processing interface

st.title("📊 SCIE Ethos — Data Processing Workflows")

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

        # Configurable folder structure (environment variables override defaults)
        self._sub = {
            "raw": tuple(os.getenv("DATA_RAW_PATH", "04_Data/00_Raw_Files").split("/")),
            "cleansed": tuple(os.getenv("DATA_CLEANSED_PATH", "04_Data/01_Cleansed_Files").split("/")),
            "eda": tuple(os.getenv("DATA_CHARTS_PATH", "04_Data/02_EDA_Charts").split("/")),
            "summaries": tuple(os.getenv("DATA_SUMMARIES_PATH", "04_Data/03_Summaries").split("/")),
            "metadata": tuple(os.getenv("DATA_METADATA_PATH", "04_Data/04_Metadata").split("/")),
            "merged": tuple(os.getenv("DATA_MERGED_PATH", "04_Data/05_Merged_Comparisons").split("/")),
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
# Ingest (debug) — Dropbox-only upload -> RAW save -> ingest -> metadata/heads
# =============================================================================
with st.expander("🔧 Ingest pipeline (debug)"):
    # 0) Detect Dropbox Project_Root once
    if "dbx_root" not in st.session_state:
        def _detect_dropbox_root(list_func):
            candidates = ["/Project_Root", "/Apps/Ethos LLM/Project_Root"]
            last_err = None
            for root in candidates:
                raw_path = "/".join([root.rstrip("/"), "04_Data", "00_Raw_Files"])
                try:
                    list_func(raw_path)
                    return root, raw_path, None
                except Exception as e:
                    last_err = str(e)
            return None, None, last_err

        detected_root, _, _ = _detect_dropbox_root(dbx_list_xlsx)
        st.session_state["dbx_root"] = detected_root or "/Project_Root"

    dbx_root = st.text_input("Project_Root (Dropbox)", value=st.session_state["dbx_root"])
    app_paths = AppPaths(project_root_dropbox=dbx_root)
    st.caption(f"RAW folder: {app_paths.dbx_raw_folder}")

    # 1) Upload (XLSX only; CSV removed per your constraint)
    st.info("📋 **Only Excel (.xlsx) files are supported. CSV files are not accepted.**")
    up = st.file_uploader(
        "Upload an Excel file (saved to Dropbox RAW, then ingested)",
        type=["xlsx", "xlsm"]
    )

    if up is not None:
        try:
            # 2) Read upload bytes
            up.seek(0)
            file_bytes = up.read()
            filename   = up.name

            # 3) Save upload to RAW in Dropbox
            raw_dest = f"{app_paths.dbx_raw_folder}/{filename}"
            upload_bytes(raw_dest, file_bytes)
            st.success(f"Saved to Dropbox RAW: {raw_dest}")

            # 4) Ingest with 5-step progress bar (calls pipeline reporter internally)
            cleaned_sheets, per_sheet_meta = ingest_streamlit_bytes_5(file_bytes, filename)

            # Save to session state for comparison engine
            st.session_state.cleaned_sheets = cleaned_sheets
            st.session_state.per_sheet_meta = per_sheet_meta

            # 5) Show per-sheet metadata summary (new pipeline returns a list)
            st.subheader("Per-sheet results")
            if per_sheet_meta:
                rows = []
                for m in per_sheet_meta:
                    rows.append({
                        "sheet_name": m.get("sheet_name"),
                        "type": m.get("normalized_sheet_type"),
                        "rows": m.get("rows"),
                        "cols": len(m.get("columns", [])) if isinstance(m.get("columns"), list) else m.get("columns"),
                        "output_path": m.get("output_path"),
                        "errors": "; ".join(m.get("errors", [])) if m.get("errors") else "",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No per-sheet metadata returned.")

            # 6) Quick peek at cleaned frames
            st.subheader("Sheets cleaned (preview)")
            st.write(list(cleaned_sheets.keys()))
            for sname, df in cleaned_sheets.items():
                st.markdown(f"### Sheet: `{sname}`")
                st.dataframe(df.head(5), use_container_width=True)

            # 7) Note: master rollups are rebuilt automatically by the pipeline
            st.caption("Master rollups refreshed in /04_Data/04_Metadata (master_*.jsonl). No manual append needed.")

        except Exception as e:
            st.error(f"Ingest (Dropbox) failed: {e}")


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
        raw_files = dbx_list_data_files(raw_dbx)  # Excel (.xlsx) only
    else:
        st.info("Switch to 'Dropbox' mode in the ingest expander to process cloud files.")
except Exception as e:
    st.error(f"Could not list RAW files: {e}")

if raw_files:
    labels = [f'{f["name"]} ({f.get("file_type", "unknown")})  ·  {f["path_lower"]}' for f in raw_files]
    # Auto-select first file (no dropdowns - auto-intent only)
    choice = labels[0] if labels else None
    
    # Quick peek
    if choice and st.button("🔎 Preview RAW file"):
        try:
            sel = raw_files[0]["path_lower"]  # Use first file directly
            file_type = raw_files[0].get("file_type", "excel")
            b = dbx_read_bytes(sel)
            
            # Handle Excel file only (CSV support removed)
            xls = pd.ExcelFile(io.BytesIO(b))
            st.success(f"Loaded Excel: {sel}")
            st.write("Sheets:", xls.sheet_names)
            if xls.sheet_names:
                first_sheet = xls.sheet_names[0]
                df_preview = pd.read_excel(io.BytesIO(b), sheet_name=first_sheet, nrows=5)
                st.caption(f"Preview: {first_sheet}")
                st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Preview failed: {e}")

    if st.button("🧹 Process & Save to Cleansed"):
        processing_start_time = time.time()  # Track processing time
        try:
            sel = raw_files[0]["path_lower"]  # Use first file directly
            filename = raw_files[0]["name"]

            # 1) Read bytes from Dropbox
            b = dbx_read_bytes(sel)

            st.info(f"🔄 Running pipeline for: {filename}")
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_container = st.container()
            
            with status_container:
                st.write("📊 **Processing Steps:**")
                step_status = st.empty()
                
                # 2) Run pipeline (bytes, filename, paths)
                step_status.write("🔄 Step 1/5: Running data cleansing pipeline...")
                progress_bar.progress(20)
                
                # Add heartbeat indicator for long-running operations
                heartbeat_container = st.container()
                with heartbeat_container:
                    heartbeat_status = st.empty()
                    heartbeat_status.write("💓 Processing... (this may take a moment)")
                
                # Phase 6: Monitor pipeline execution
                pipeline_start_time = time.time()
                if PHASE6_AVAILABLE and monitoring:
                    monitoring.increment("pipeline.executions", 1, pipeline="phase1_ingest")
                    monitoring.gauge("pipeline.file_size", len(b), filename=filename)
                
                # Check cache for existing results
                cache_key = None
                if PHASE6_AVAILABLE and cache:
                    import hashlib
                    file_hash = hashlib.md5(b).hexdigest()
                    cache_key = f"pipeline:{file_hash}:{filename}"
                    cached_result = cache.get(cache_key)
                    if cached_result:
                        cleaned_sheets, metadata = cached_result
                        st.success("🚀 Retrieved results from cache!")
                        if monitoring:
                            monitoring.increment("cache.hits", 1, type="pipeline_results")
                    else:
                        cleaned_sheets, per_sheet_meta = run_pipeline(b, filename, app_paths)
                        
                        # Convert per_sheet_meta list to proper metadata dictionary structure
                        metadata = {
                            "source_filename": filename,
                            "sheet_count": len(per_sheet_meta),
                            "sheets": per_sheet_meta,
                            "processing_timestamp": time.time(),
                            "pipeline_version": "phase1_ingest"
                        }
                        
                        # Cache the results
                        cache.set(cache_key, (cleaned_sheets, metadata), ttl=3600, tags=["pipeline", "results"])
                        if monitoring:
                            monitoring.increment("cache.misses", 1, type="pipeline_results")
                else:
                    cleaned_sheets, per_sheet_meta = run_pipeline(b, filename, app_paths)
                    
                    # Convert per_sheet_meta list to proper metadata dictionary structure
                    metadata = {
                        "source_filename": filename,
                        "sheet_count": len(per_sheet_meta),
                        "sheets": per_sheet_meta,
                        "processing_timestamp": time.time(),
                        "pipeline_version": "phase1_ingest"
                    }
                
                # Save to session state for comparison engine
                st.session_state.cleaned_sheets = cleaned_sheets
                st.session_state.per_sheet_meta = per_sheet_meta
                
                # Phase 6: Monitor pipeline completion
                if PHASE6_AVAILABLE and monitoring:
                    pipeline_execution_time = time.time() - pipeline_start_time
                    monitoring.timing("pipeline.execution_time", pipeline_execution_time, pipeline="phase1_ingest")
                    monitoring.gauge("pipeline.sheets_processed", len(cleaned_sheets), filename=filename)
                
                # Clear heartbeat
                heartbeat_status.empty()
                
                step_status.write("✅ Step 1/5: Data cleansing completed")
                progress_bar.progress(40)

            # 3) Enhanced EDA Generation and Display (Comprehensive Colab-style workflow)
            st.subheader("📈 Comprehensive Data Analysis & Insights")
            
            # Import enhanced EDA tools matching Colab workflow
            try:
                from phase2_analysis.enhanced_eda_system import run_enhanced_eda
                from phase2_analysis.gpt_summary_generator import generate_comprehensive_summary
                from phase2_analysis.smart_autofix_system import run_smart_autofix
                enhanced_eda_available = True
                st.success("✅ Enhanced EDA system loaded (Colab-style workflow)")
            except ImportError as e:
                st.warning(f"⚠️ Enhanced EDA modules not available: {e}")
                enhanced_eda_available = False

            all_insights = {}
            all_chart_paths = []
            all_eda_results = {}
            
            for sheet_name, df in cleaned_sheets.items():
                with st.expander(f"📊 Comprehensive Analysis: {sheet_name} ({len(df)} rows, {len(df.columns)} columns)", expanded=True):
                    
                    # Basic stats display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        missing_pct = round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
                        st.metric("Missing Data", f"{missing_pct}%")
                    with col4:
                        data_quality_score = round(100 - missing_pct, 1)
                        st.metric("Quality Score", f"{data_quality_score}%")
                    
                    # Show sample data
                    st.write("**Sample Data Preview:**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # STEP 1: Smart Auto-Fix (matching Colab workflow)
                    if enhanced_eda_available:
                        st.write("---")
                        st.write("🧹 **STEP 1: Smart Auto-Fix & Data Cleaning**")
                        
                        with st.spinner("Running intelligent auto-fix system..."):
                            try:
                                # Run comprehensive auto-fix
                                df_cleaned, cleaning_operations, cleaning_report = run_smart_autofix(
                                    df.copy(), sheet_name, aggressive_mode=False
                                )
                                
                                if len(cleaning_operations) > 0:
                                    st.success(f"✅ Applied {len(cleaning_operations)} auto-fix operations")
                                    
                                    # Show key improvements
                                    improvement_col1, improvement_col2 = st.columns(2)
                                    with improvement_col1:
                                        original_missing = df.isnull().sum().sum()
                                        cleaned_missing = df_cleaned.isnull().sum().sum()
                                        missing_reduction = original_missing - cleaned_missing
                                        st.metric("Missing Values Reduced", f"{missing_reduction:,}")
                                    
                                    with improvement_col2:
                                        original_quality = 100 - (original_missing / (len(df) * len(df.columns))) * 100
                                        cleaned_quality = 100 - (cleaned_missing / (len(df_cleaned) * len(df_cleaned.columns))) * 100
                                        quality_improvement = cleaned_quality - original_quality
                                        st.metric("Quality Improvement", f"+{quality_improvement:.1f}%")
                                    
                                    # Show cleaning operations summary
                                    with st.expander("🔍 View Auto-Fix Operations"):
                                        for i, operation in enumerate(cleaning_operations[:10], 1):
                                            st.write(f"{i}. {operation}")
                                        if len(cleaning_operations) > 10:
                                            st.write(f"... and {len(cleaning_operations) - 10} more operations")
                                    
                                    # Use cleaned data for further analysis
                                    df = df_cleaned
                                    
                                else:
                                    st.info("ℹ️ No auto-fix operations needed - data quality is good")
                                    
                            except Exception as e:
                                st.warning(f"⚠️ Auto-fix failed: {e}")
                    
                    # STEP 2: Comprehensive EDA (matching Colab workflow)
                    if enhanced_eda_available:
                        st.write("---")
                        st.write("📊 **STEP 2: Comprehensive Multi-Round EDA**")
                        
                        with st.spinner("Running comprehensive EDA analysis..."):
                            try:
                                # Phase 6: Monitor EDA execution
                                eda_start_time = time.time()
                                if PHASE6_AVAILABLE and monitoring:
                                    monitoring.increment("eda.executions", 1, sheet=sheet_name, filename=filename)
                                
                                # Run enhanced EDA system
                                eda_results = run_enhanced_eda(
                                    df, 
                                    sheet_name=sheet_name,
                                    filename=filename,
                                    charts_folder=getattr(app_paths, "dbx_eda_folder", "/Project_Root/04_Data/02_EDA_Charts"),
                                    summaries_folder=getattr(app_paths, "dbx_summaries_folder", "/Project_Root/04_Data/03_Summaries"),
                                    max_rounds=3
                                )
                                
                                # Phase 6: Monitor EDA completion
                                if PHASE6_AVAILABLE and monitoring:
                                    eda_execution_time = time.time() - eda_start_time
                                    monitoring.timing("eda.execution_time", eda_execution_time, sheet=sheet_name, filename=filename)
                                    monitoring.gauge("eda.charts_generated", len(eda_results.get("chart_paths", [])), sheet=sheet_name)
                                
                                all_eda_results[sheet_name] = eda_results
                                chart_paths = eda_results.get("chart_paths", [])
                                all_chart_paths.extend(chart_paths)
                                
                                # Display EDA results summary
                                rounds_completed = len(eda_results.get("rounds", []))
                                charts_generated = len(chart_paths)
                                
                                eda_col1, eda_col2, eda_col3 = st.columns(3)
                                with eda_col1:
                                    st.metric("Analysis Rounds", rounds_completed)
                                with eda_col2:
                                    st.metric("Charts Generated", charts_generated)
                                with eda_col3:
                                    business_insights = eda_results.get("business_insights", {})
                                    supply_metrics = len(business_insights.get("supply_chain_metrics", {}))
                                    st.metric("Supply Chain Metrics", supply_metrics)
                                
                                # Display business insights
                                supply_chain_metrics = business_insights.get("supply_chain_metrics", {})
                                if supply_chain_metrics:
                                    st.write("**📦 Supply Chain Financial Overview:**")
                                    for category, metrics in supply_chain_metrics.items():
                                        if metrics.get("total_value", 0) > 0:
                                            st.write(f"- **{category.title()}**: ${metrics['total_value']:,.2f} across {metrics.get('column_count', 0)} columns")
                                
                            except Exception as e:
                                st.error(f"❌ Enhanced EDA failed: {e}")
                                # Fallback to basic analysis
                                enhanced_eda_available = False
                    
                    # Fallback to basic analysis if enhanced EDA not available
                    
                    if enhanced_eda_available and len(df) > 0:
                        # Generate suggested EDA actions
                        step_status.write(f"🔄 Step 2/5: Generating visualizations for {sheet_name}...")
                        progress_bar.progress(60)
                        
                        # Create EDA actions based on data types
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        suggested_actions = []
                        
                        # Add histograms for numeric columns (top 3)
                        for col in numeric_cols[:3]:
                            suggested_actions.append({"action": "histogram", "column": col})
                        
                        # Add scatter plots for numeric pairs
                        if len(numeric_cols) >= 2:
                            suggested_actions.append({
                                "action": "scatter", 
                                "x": numeric_cols[0], 
                                "y": numeric_cols[1]
                            })
                        
                        # Add correlation matrix if enough numeric columns
                        if len(numeric_cols) >= 3:
                            suggested_actions.append({"action": "correlation_matrix"})
                        
                        # Add top N analysis for categorical + numeric
                        if categorical_cols and numeric_cols:
                            suggested_actions.append({
                                "action": "groupby_topn",
                                "group": categorical_cols[0],
                                "metric": numeric_cols[0],
                                "topn": 10
                            })
                        
                        # Supply chain dashboard if relevant columns found
                        supply_keywords = ['wip', 'inventory', 'cost', 'value', 'qty', 'quantity']
                        if any(keyword in ' '.join(df.columns).lower() for keyword in supply_keywords):
                            suggested_actions.append({
                                "action": "supply_chain_dashboard",
                                "title": f"Supply Chain Analysis: {sheet_name}"
                            })
                        
                        # Generate charts - Use Dropbox-compatible approach
                        charts_folder = getattr(app_paths, "dbx_eda_folder", None)
                        if not charts_folder:
                            # Default to metadata folder since we know that exists
                            charts_folder = f"{getattr(app_paths, 'dbx_metadata_folder', '/Project_Root/04_Data/04_Metadata')}"
                        
                        suffix = f"_{sheet_name.replace(' ', '_').replace('/', '_')}"
                        
                        # Use cloud-safe chart generation
                        chart_paths = []
                        charts_generated = 0
                        
                        try:
                            # Instead of using run_gpt_eda_actions which tries to create local dirs,
                            # let's create charts using cloud-compatible methods
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from io import BytesIO
                            
                            # Set style for better looking charts
                            plt.style.use('default')
                            sns.set_palette("husl")
                            
                            # Create a progress container for chart generation
                            chart_progress_container = st.container()
                            with chart_progress_container:
                                chart_status = st.empty()
                                chart_progress = st.progress(0)
                                total_charts = min(3, len(numeric_cols)) + (1 if len(numeric_cols) >= 2 else 0) + (1 if len(numeric_cols) >= 3 else 0) + (1 if categorical_cols and numeric_cols else 0)
                                current_chart = 0
                            
                            # Generate histograms for numeric columns (top 3)
                            for i, col in enumerate(numeric_cols[:3]):
                                if df[col].notna().sum() > 0:  # Only if we have data
                                    try:
                                        chart_status.write(f"🎨 Creating histogram for {col}...")
                                        
                                        fig, ax = plt.subplots(figsize=(8, 4))  # Smaller size: was (10, 6)
                                        
                                        # Create histogram
                                        df[col].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
                                        ax.set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')  # Smaller font
                                        ax.set_xlabel(col, fontsize=10)  # Smaller font
                                        ax.set_ylabel('Frequency', fontsize=10)  # Smaller font
                                        ax.grid(True, alpha=0.3)
                                        
                                        # Save to Dropbox via bytes
                                        chart_buffer = BytesIO()
                                        plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
                                        chart_buffer.seek(0)
                                        
                                        chart_path = f"{charts_folder}/hist_{col.replace(' ', '_')}{suffix}.png"
                                        upload_bytes(chart_path, chart_buffer.getvalue())
                                        chart_paths.append(chart_path)
                                        charts_generated += 1
                                        current_chart += 1
                                        
                                        chart_progress.progress(current_chart / total_charts)
                                        chart_status.write(f"✅ Histogram for {col} completed")
                                        
                                        plt.close(fig)
                                        
                                    except Exception as chart_e:
                                        st.warning(f"Could not generate histogram for {col}: {chart_e}")
                            
                            # Generate scatter plot for first two numeric columns
                            if len(numeric_cols) >= 2:
                                try:
                                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                                    chart_status.write(f"🎨 Creating scatter plot: {x_col} vs {y_col}...")
                                    clean_data = df[[x_col, y_col]].dropna()
                                    
                                    if len(clean_data) > 0:
                                        fig, ax = plt.subplots(figsize=(8, 4))  # Smaller size: was (10, 6)
                                        
                                        ax.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, color='coral')
                                        ax.set_title(f'Relationship: {x_col} vs {y_col}', fontsize=12, fontweight='bold')  # Smaller font
                                        ax.set_xlabel(x_col, fontsize=10)  # Smaller font
                                        ax.set_ylabel(y_col, fontsize=10)  # Smaller font
                                        ax.grid(True, alpha=0.3)
                                        
                                        # Save to Dropbox
                                        chart_buffer = BytesIO()
                                        plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
                                        chart_buffer.seek(0)
                                        
                                        chart_path = f"{charts_folder}/scatter_{x_col.replace(' ', '_')}_vs_{y_col.replace(' ', '_')}{suffix}.png"
                                        upload_bytes(chart_path, chart_buffer.getvalue())
                                        chart_paths.append(chart_path)
                                        charts_generated += 1
                                        current_chart += 1
                                        
                                        chart_progress.progress(current_chart / total_charts)
                                        chart_status.write(f"✅ Scatter plot completed")
                                        
                                        plt.close(fig)
                                        
                                except Exception as chart_e:
                                    st.warning(f"Could not generate scatter plot: {chart_e}")
                            
                            # Generate correlation heatmap if enough numeric columns
                            if len(numeric_cols) >= 3:
                                try:
                                    chart_status.write("🎨 Creating correlation heatmap...")
                                    corr_data = df[numeric_cols].corr()
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))  # Smaller size: was (12, 8)
                                    
                                    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', 
                                              center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
                                    ax.set_title('Correlation Analysis', fontsize=14, fontweight='bold', pad=15)  # Smaller padding
                                    
                                    # Save to Dropbox
                                    chart_buffer = BytesIO()
                                    plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
                                    chart_buffer.seek(0)
                                    
                                    chart_path = f"{charts_folder}/correlation_heatmap{suffix}.png"
                                    upload_bytes(chart_path, chart_buffer.getvalue())
                                    chart_paths.append(chart_path)
                                    charts_generated += 1
                                    current_chart += 1
                                    
                                    chart_progress.progress(current_chart / total_charts)
                                    chart_status.write(f"✅ Correlation heatmap completed")
                                    
                                    plt.close(fig)
                                    
                                except Exception as chart_e:
                                    st.warning(f"Could not generate correlation heatmap: {chart_e}")
                            
                            # Generate top N analysis if we have categorical and numeric
                            if categorical_cols and numeric_cols:
                                try:
                                    group_col, metric_col = categorical_cols[0], numeric_cols[0]
                                    chart_status.write(f"🎨 Creating top 10 analysis: {group_col} by {metric_col}...")
                                    top_data = df.groupby(group_col)[metric_col].sum().sort_values(ascending=False).head(10)
                                    
                                    if len(top_data) > 0:
                                        fig, ax = plt.subplots(figsize=(10, 4))  # Smaller size: was (12, 6)
                                        
                                        top_data.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
                                        ax.set_title(f'Top 10 {group_col} by {metric_col}', fontsize=12, fontweight='bold')  # Smaller font
                                        ax.set_xlabel(group_col, fontsize=10)  # Smaller font
                                        ax.set_ylabel(metric_col, fontsize=10)  # Smaller font
                                        ax.tick_params(axis='x', rotation=45)
                                        ax.grid(True, alpha=0.3)
                                        
                                        # Save to Dropbox
                                        chart_buffer = BytesIO()
                                        plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
                                        chart_buffer.seek(0)
                                        
                                        chart_path = f"{charts_folder}/top10_{group_col.replace(' ', '_')}_by_{metric_col.replace(' ', '_')}{suffix}.png"
                                        upload_bytes(chart_path, chart_buffer.getvalue())
                                        chart_paths.append(chart_path)
                                        charts_generated += 1
                                        current_chart += 1
                                        
                                        chart_progress.progress(1.0)  # Complete
                                        chart_status.write(f"✅ All visualizations completed!")
                                        
                                        plt.close(fig)
                                        
                                except Exception as chart_e:
                                    st.warning(f"Could not generate top N analysis: {chart_e}")
                            
                            # Clear the progress indicators after completion
                            time.sleep(1)  # Brief pause to show completion
                            chart_status.empty()
                            chart_progress.empty()
                            
                        except ImportError:
                            st.warning("⚠️ Matplotlib/Seaborn not available. Skipping chart generation.")
                            chart_paths = []
                            charts_generated = 0
                        
                        except Exception as e:
                            st.warning(f"⚠️ Chart generation failed: {e}")
                            chart_paths = []
                            charts_generated = 0
                        
                        all_chart_paths.extend(chart_paths)
                        
                        # Create basic enhanced summary
                        enhanced_summary = {
                            "dataset_overview": {
                                "total_rows": len(df),
                                "total_columns": len(df.columns),
                                "missing_data_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
                            },
                            "charts_generated": chart_paths,
                            "charts_folder": charts_folder
                        }
                        
                        # Add supply chain insights
                        supply_chain_cols = {
                            'wip': [col for col in df.columns if 'wip' in str(col).lower()],
                            'cost': [col for col in df.columns if any(term in str(col).lower() for term in ['cost', 'value', 'price'])],
                            'quantity': [col for col in df.columns if any(term in str(col).lower() for term in ['qty', 'quantity', 'count'])]
                        }
                        
                        enhanced_summary["supply_chain_insights"] = {}
                        for category, cols in supply_chain_cols.items():
                            if cols:
                                try:
                                    total_value = float(df[cols].sum().sum()) if cols else 0
                                    enhanced_summary["supply_chain_insights"][category] = {
                                        "columns_found": cols,
                                        "total_value": total_value
                                    }
                                except:
                                    enhanced_summary["supply_chain_insights"][category] = {
                                        "columns_found": cols,
                                        "total_value": 0
                                    }
                        
                        all_insights[sheet_name] = enhanced_summary
                        
                        # Display key insights
                        if enhanced_summary.get("supply_chain_insights"):
                            st.write("**📦 Supply Chain Insights:**")
                            for category, info in enhanced_summary["supply_chain_insights"].items():
                                if info.get("total_value", 0) > 0:
                                    st.write(f"- **{category.title()}**: ${info['total_value']:,.2f} across {len(info['columns_found'])} columns")
                        
                        # Show chart generation results with display and progress
                        if chart_paths:
                            st.success(f"✅ Generated {charts_generated} visualizations")
                            
                            # Display charts inline in a compact grid layout
                            st.write("**📊 Generated Visualizations:**")
                            
                            chart_display_container = st.container()
                            with chart_display_container:
                                # Group charts in rows of 2
                                for i in range(0, len(chart_paths), 2):
                                    # Create columns for side-by-side display
                                    if i + 1 < len(chart_paths):
                                        col1, col2 = st.columns(2)
                                        charts_in_row = [chart_paths[i], chart_paths[i + 1]]
                                        columns = [col1, col2]
                                    else:
                                        col1, col2 = st.columns(2)
                                        charts_in_row = [chart_paths[i]]
                                        columns = [col1]
                                    
                                    # Display charts in this row
                                    for j, (path, col) in enumerate(zip(charts_in_row, columns)):
                                        with col:
                                            chart_name = path.split('/')[-1]
                                            st.caption(f"� {chart_name}")
                                            
                                            # Try to display the chart by reading it back from Dropbox
                                            try:
                                                chart_bytes = dbx_read_bytes(path)
                                                st.image(chart_bytes, caption=None, use_container_width=True)
                                            except Exception as display_e:
                                                st.caption(f"⚠️ Chart saved but display failed: {display_e}")
                                    
                                    # Add spacing between rows
                                    if i + 2 < len(chart_paths):
                                        st.write("")
                                        
                        else:
                            st.info("ℹ️ No charts generated (insufficient data or columns)")
                    
                    else:
                        # Basic EDA fallback
                        basic_summary = generate_eda_summary(df) if 'generate_eda_summary' in globals() else {}
                        if basic_summary:
                            st.write("**Basic Statistics:**")
                            if "descriptive_stats" in basic_summary:
                                st.dataframe(pd.DataFrame(basic_summary["descriptive_stats"]).T)

            # STEP 3: Generate Comprehensive AI Summaries (matching Colab workflow)
            if enhanced_eda_available:
                step_status.write("🔄 Step 3/5: Generating comprehensive AI summaries...")
                progress_bar.progress(75)
                
                try:
                    # Generate comprehensive summaries for all sheets
                    comprehensive_summaries = {}
                    
                    for sheet_name, df in cleaned_sheets.items():
                        sheet_metadata = metadata.get("sheets", [])
                        sheet_meta = next((s for s in sheet_metadata if s.get("sheet_name") == sheet_name), {})
                        
                        sheet_eda_results = all_eda_results.get(sheet_name, {})
                        
                        summaries = generate_comprehensive_summary(
                            df=df,
                            sheet_name=sheet_name,
                            filename=filename,
                            metadata=sheet_meta,
                            eda_results=sheet_eda_results
                        )
                        
                        comprehensive_summaries[sheet_name] = summaries
                    
                    # Store summaries in metadata
                    metadata["comprehensive_summaries"] = comprehensive_summaries
                    
                    st.success("✅ AI-powered comprehensive summaries generated")
                    
                except Exception as e:
                    st.warning(f"⚠️ AI summary generation failed: {e}")
            
            step_status.write("✅ Step 2/5: Comprehensive analysis completed")
            progress_bar.progress(80)

            # 4) Save cleansed workbook back to Dropbox
            step_status.write("🔄 Step 4/5: Saving cleansed workbook...")
            out_bytes = build_xlsx_bytes_from_sheets(cleaned_sheets)
            out_base = filename.rsplit(".xlsx", 1)[0]
            cleansed_dbx = getattr(app_paths, "dbx_cleansed_folder", None)
            if not cleansed_dbx:
                raise RuntimeError("Dropbox cleansed folder not set (switch to 'Dropbox' mode).")
            out_path = f"{cleansed_dbx}/{out_base}_cleansed.xlsx"
            upload_bytes(out_path, out_bytes)
            step_status.write("✅ Step 4/5: Cleansed workbook saved")
            progress_bar.progress(85)

            # 5) Update master metadata on Dropbox
            step_status.write("🔄 Step 5/5: Updating metadata...")
            progress_bar.progress(85)
            
            # Add comprehensive results to metadata
            if enhanced_eda_available:
                metadata["enhanced_eda_results"] = all_eda_results
                metadata["total_charts_generated"] = len(all_chart_paths)
            else:
                metadata["eda_insights"] = all_insights
                metadata["chart_paths"] = all_chart_paths
            
            meta_path = getattr(app_paths, "dbx_master_metadata_path", None) or getattr(app_paths, "master_metadata_path", None)
            if not meta_path:
                raise RuntimeError("No metadata path available (neither Dropbox nor local).")
            
            # Add retry logic for network issues
            metadata_updated = False
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    step_status.write(f"🔄 Step 5/5: Updating metadata (attempt {attempt + 1}/{max_retries})...")
                    
                    existing = []
                    try:
                        existing = json.loads(dbx_read_bytes(meta_path).decode("utf-8"))
                        if not isinstance(existing, list):
                            existing = []
                    except Exception:
                        existing = []
                    
                    existing.append(metadata)
                    upload_json(meta_path, existing)
                    metadata_updated = True
                    step_status.write("✅ Step 5/5: Metadata updated successfully")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        step_status.write(f"⚠️ Metadata update attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)  # Wait before retry
                    else:
                        st.warning(f"⚠️ Metadata update failed after {max_retries} attempts: {e}")
                        st.info("📊 Data processing completed successfully, but metadata sync had network issues.")
            
            progress_bar.progress(95)

            # 6) Save per-file summaries to Dropbox (JSON + Markdown)
            step_status.write("🔄 Step 6/6: Saving comprehensive summaries...")
            progress_bar.progress(95)
            
            summaries_dbx = getattr(app_paths, "dbx_summaries_folder", None)
            if summaries_dbx:
                summary_saved = False
                
                # Save run metadata with retry
                for attempt in range(max_retries):
                    try:
                        step_status.write(f"🔄 Step 6/6: Saving metadata summary (attempt {attempt + 1}/{max_retries})...")
                        run_meta_path = f"{summaries_dbx}/{out_base}_run_metadata.json"
                        upload_json(run_meta_path, metadata)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                        else:
                            st.warning(f"⚠️ Metadata summary save failed: {e}")
                
                # Save executive summary with retry  
                for attempt in range(max_retries):
                    try:
                        step_status.write(f"🔄 Step 6/6: Saving executive summary (attempt {attempt + 1}/{max_retries})...")
                        exec_md_bytes = _build_summary_markdown(metadata).encode("utf-8")
                        exec_md_path = f"{summaries_dbx}/{out_base}_executive_summary.md"
                        upload_bytes(exec_md_path, exec_md_bytes)
                        summary_saved = True
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                        else:
                            st.warning(f"⚠️ Executive summary save failed: {e}")

                step_status.write("✅ Step 6/6: All processing completed!")
                progress_bar.progress(100)
                
                # Final success message
                st.success("🎉 **Processing Complete!**")
                
                # Create final summary with better formatting
                success_info = f"""
                **📊 Files Created:**
                - **Cleansed Data**: `{out_path}`
                - **Charts Generated**: {len(all_chart_paths)} visualizations 
                - **Metadata**: `{meta_path}` {"✅" if metadata_updated else "⚠️"}
                """
                
                if summary_saved:
                    success_info += f"- **Summary**: `{exec_md_path}` ✅"
                
                st.info(success_info)
                
                # Show processing time if available
                processing_time = time.time() - processing_start_time
                st.caption(f"⏱️ Total processing time: {processing_time:.1f} seconds")
                
            else:
                st.warning("Summaries folder not set on Dropbox (03_Summaries). Skipping summaries save.")
                step_status.write("✅ Step 6/6: Processing completed (summaries skipped)")
                progress_bar.progress(100)

            # Display comprehensive insights summary (enhanced for Colab-style results)
            if enhanced_eda_available and all_eda_results:
                st.subheader("🎯 Comprehensive Analysis Results")
                
                total_rows = sum(len(df) for df in cleaned_sheets.values())
                total_charts = len(all_chart_paths)
                total_rounds = sum(len(results.get("rounds", [])) for results in all_eda_results.values())
                
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                with result_col1:
                    st.metric("Records Processed", f"{total_rows:,}")
                with result_col2:
                    st.metric("Sheets Analyzed", len(cleaned_sheets))
                with result_col3:
                    st.metric("Analysis Rounds", total_rounds)
                with result_col4:
                    st.metric("Charts Generated", total_charts)
                
                # Show business insights from enhanced analysis
                st.write("**🎯 Key Business Insights:**")
                for sheet_name, eda_results in all_eda_results.items():
                    business_insights = eda_results.get("business_insights", {})
                    supply_metrics = business_insights.get("supply_chain_metrics", {})
                    
                    if supply_metrics:
                        st.write(f"**{sheet_name}:**")
                        for category, metrics in supply_metrics.items():
                            if metrics.get("total_value", 0) > 0:
                                st.write(f"  - {category.title()}: ${metrics['total_value']:,.2f}")
                
            elif all_insights:
                st.subheader("🎯 Final Insights Summary")
                total_rows = sum(len(df) for df in cleaned_sheets.values())
                total_charts = len(all_chart_paths)
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                with insight_col1:
                    st.metric("Total Records Processed", f"{total_rows:,}")
                with insight_col2:
                    st.metric("Sheets Analyzed", len(cleaned_sheets))
                with insight_col3:
                    st.metric("Charts Generated", total_charts)

            # Display Enhanced Executive Summary (Colab-style)
            st.subheader("📋 Executive Summary")
            try:
                if enhanced_eda_available and metadata.get("comprehensive_summaries"):
                    # Display AI-generated comprehensive summaries
                    st.write("**🧠 AI-Powered Analysis Results:**")
                    
                    for sheet_name, summaries in metadata["comprehensive_summaries"].items():
                        with st.expander(f"� {sheet_name} - Comprehensive Analysis", expanded=True):
                            
                            executive_summary = summaries.get("executive_summary", "")
                            if executive_summary:
                                st.markdown("### Executive Summary")
                                st.write(executive_summary)
                                st.write("")
                            
                            data_quality_report = summaries.get("data_quality_report", "")
                            if data_quality_report:
                                st.markdown("### Data Quality Assessment")
                                st.write(data_quality_report)
                                st.write("")
                            
                            eda_insights = summaries.get("eda_insights", "")
                            if eda_insights:
                                st.markdown("### EDA Insights")
                                st.write(eda_insights)
                
                else:
                    # Fallback to generate and display the basic executive summary
                    summary_markdown = _build_summary_markdown(metadata)
                    st.markdown(summary_markdown)
                
                # Also show key insights in a more digestible format
                st.subheader("🔍 Key Insights by Sheet")
                
                if "sheets" in metadata:
                    for sheet_info in metadata["sheets"]:
                        sheet_name = sheet_info.get("sheet_name", "Unknown")
                        sheet_type = sheet_info.get("normalized_sheet_type", "Unknown")
                        record_count = sheet_info.get("rows", 0)
                        summary_text = sheet_info.get("summary_text", "No summary available")
                        eda_text = sheet_info.get("eda_text", "")
                        
                        with st.expander(f"📊 {sheet_name} ({sheet_type}) - {record_count:,} records"):
                            if summary_text and summary_text != "No summary available":
                                st.write("**📄 Summary:**")
                                st.write(summary_text)
                                st.write("")  # Add spacing
                            
                            # Show supply chain insights if available - FIRST, as these are most important
                            sheet_insights = None
                            if enhanced_eda_available and all_eda_results.get(sheet_name):
                                sheet_insights = all_eda_results[sheet_name].get("business_insights", {})
                            elif all_insights.get(sheet_name):
                                sheet_insights = all_insights[sheet_name]
                            
                            if sheet_insights and sheet_insights.get("supply_chain_insights"):
                                st.write("**📦 Supply Chain Metrics:**")
                                for category, info in sheet_insights["supply_chain_insights"].items():
                                    if info.get("total_value", 0) > 0:
                                        st.write(f"- **{category.title()}**: ${info['total_value']:,.2f}")
                                st.write("")  # Add spacing
                            
                            # Show dataset overview in a cleaner format
                            if sheet_insights and sheet_insights.get("dataset_overview"):
                                overview = sheet_insights["dataset_overview"]
                                st.write("**📈 Data Quality Overview:**")
                                
                                qual_col1, qual_col2, qual_col3 = st.columns(3)
                                with qual_col1:
                                    st.metric("Total Records", f"{overview.get('total_rows', 0):,}")
                                with qual_col2:
                                    st.metric("Columns", overview.get('total_columns', 0))
                                with qual_col3:
                                    missing_pct = overview.get('missing_data_percentage', 0)
                                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                                
                                chart_count = len(sheet_insights.get("charts_generated", []))
                                if chart_count > 0:
                                    st.write(f"📊 **{chart_count} visualizations generated** for this sheet")
                                st.write("")  # Add spacing
                            
                            # Enhanced EDA results display
                            if enhanced_eda_available and sheet_name in all_eda_results:
                                eda_result = all_eda_results[sheet_name]
                                final_summary = eda_result.get("final_summary", {})
                                
                                if final_summary.get("summary_text"):
                                    st.write("**🔍 AI Analysis Summary:**")
                                    st.write(final_summary["summary_text"])
                                    st.write("")
                                
                                # Show round-by-round insights
                                rounds = eda_result.get("rounds", [])
                                if rounds:
                                    st.write("**📊 Analysis Rounds:**")
                                    for round_data in rounds:
                                        round_num = round_data.get("round_number", 0)
                                        analysis_type = round_data.get("analysis_type", "")
                                        chart_count = len(round_data.get("chart_paths", []))
                                        ai_summary = round_data.get("ai_summary", "")
                                        
                                        st.write(f"- **Round {round_num}** ({analysis_type}): {chart_count} charts")
                                        if ai_summary:
                                            st.write(f"  {ai_summary}")
                            
                            # Only show basic EDA text if enhanced results not available and actually useful
                            elif eda_text and eda_text.strip() and not eda_text.startswith('{'):
                                st.write("**🔍 Detailed Analysis:**")
                                st.write(eda_text)
                            elif sheet_insights:
                                # Generate a more meaningful analysis summary
                                st.write("**🔍 Key Data Insights:**")
                                
                                # If we have supply chain data, focus on that
                                if sheet_insights.get("supply_chain_insights"):
                                    supply_insights = sheet_insights["supply_chain_insights"]
                                    meaningful_categories = [cat for cat, info in supply_insights.items() 
                                                           if info.get("total_value", 0) > 0]
                                    
                                    if meaningful_categories:
                                        st.write(f"This sheet contains **{sheet_type}** data with {len(meaningful_categories)} key financial metrics:")
                                        for category in meaningful_categories:
                                            info = supply_insights[category]
                                            cols = info.get("columns_found", [])
                                            value = info.get("total_value", 0)
                                            st.write(f"- **{category.title()}**: ${value:,.2f} across columns: {', '.join(cols[:3])}")
                                
                                # Add data quality insights
                                overview = sheet_insights.get("dataset_overview", {})
                                missing_pct = overview.get('missing_data_percentage', 0)
                                if missing_pct > 10:
                                    st.warning(f"⚠️ High missing data rate ({missing_pct:.1f}%) - may need data cleaning")
                                elif missing_pct > 0:
                                    st.info(f"ℹ️ Low missing data rate ({missing_pct:.1f}%) - good data quality")
                                else:
                                    st.success("✅ No missing data - excellent data quality")
                
            except Exception as e:
                st.warning(f"Could not display executive summary: {e}")

            with st.expander("📋 View Raw Metadata"):
                st.json(metadata)

        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            st.write("**Debug Info:**")
            st.write(f"- Error type: {type(e).__name__}")
            st.write(f"- Error details: {str(e)}")
            
            # Show stack trace in expander for debugging
            import traceback
            with st.expander("🔧 Stack Trace (for debugging)"):
                st.code(traceback.format_exc())
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
        cln_files = dbx_list_data_files(cln_dbx)  # Excel (.xlsx) only
    else:
        st.info("Switch to 'Dropbox' mode to browse Cleansed files.")
except Exception as e:
    st.error(f"Could not list Cleansed files: {e}")

if cln_files:
    cln_labels = [f'{f["name"]} ({f.get("file_type", "unknown")})  ·  {f["path_lower"]}' for f in cln_files]
    # Auto-select first file (no dropdowns - auto-intent only)
    cln_choice = cln_labels[0] if cln_labels else None
    if cln_choice and st.button("🔎 Preview Cleansed file"):
        try:
            sel = cln_files[0]["path_lower"]  # Use first file directly
            file_type = cln_files[0].get("file_type", "excel")
            b = dbx_read_bytes(sel)
            
            # Handle Excel file only (CSV support removed)
            xls = pd.ExcelFile(io.BytesIO(b))
            st.success(f"Loaded Excel: {sel}")
            st.write("Sheets:", xls.sheet_names)
            if xls.sheet_names:
                first_sheet = xls.sheet_names[0]
                df_preview = pd.read_excel(io.BytesIO(b), sheet_name=first_sheet, nrows=5)
                st.caption(f"Preview: {first_sheet}")
                df_preview = pd.read_excel(io.BytesIO(b), sheet_name=first_sheet, nrows=5)
                st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Preview failed: {e}")
else:
    st.info("No files in Cleansed yet. Run step 1 first (Dropbox mode).")

# =============================================================================
# Knowledge Base (optional, cloud-aware)
# =============================================================================
st.header("3) Knowledge Base (optional)")
# Use the computed cloud-aware PROJECT_ROOT from knowledgebase_builder
PROJECT_ROOT = KB_PROJECT_ROOT
with st.expander("Knowledge Base controls"):
    st.json(kb_status(PROJECT_ROOT))
    include_text = st.checkbox("Include .txt/.md", value=True)
    force_rebuild = st.checkbox("Force full rebuild", value=False)
    if st.button("🔧 Build / Update KB"):
            with st.spinner("Building knowledge base..."):
                # Phase 6: Monitor KB build
                kb_start_time = time.time()
                if PHASE6_AVAILABLE and monitoring:
                    monitoring.increment("kb.builds", 1, force_rebuild=force_rebuild)
                
                res = build_or_update_knowledgebase(
                    project_root=PROJECT_ROOT,
                    scan_folders=None,
                    force_rebuild=force_rebuild,
                    include_text_files=include_text
                )
                
                # Phase 6: Monitor KB build completion
                if PHASE6_AVAILABLE and monitoring:
                    kb_build_time = time.time() - kb_start_time
                    monitoring.timing("kb.build_time", kb_build_time, force_rebuild=force_rebuild)
                    monitoring.gauge("kb.files_processed", res.get("files_processed", 0), project_root=PROJECT_ROOT)
                
                st.success("KB build complete")
                st.json(res)

# =========================
# 4) Ask questions about a Cleansed workbook (cloud-only)
# =========================
import datetime as _dt
from orchestrator import answer_question
from dbx_utils import list_data_files

st.markdown("## 4) Ask questions about a Cleansed workbook")

# Use the same path construction as the working section (line 1390)
cleansed_folder = getattr(app_paths, "dbx_cleansed_folder", None)
print(f"DEBUG: Using cleansed folder: {cleansed_folder}")

if not cleansed_folder:
    st.warning("Cleansed folder not configured on Dropbox. Set app_paths.dbx_cleansed_folder.")
else:
    try:
        files = list_data_files(cleansed_folder)  # Excel (.xlsx) only
    except Exception as e:
        files = []
        st.error(f"Could not list cleansed files: {e}")

    if not files:
        st.info("No cleansed files found in Dropbox yet. Run a cleanse in section 1 first.")
    else:
        # Recent first (list_data_files already sorts newest first)
        nice_labels = [
            f"{f['name']} ({f.get('file_type', 'unknown')})"
            + (f" — {f['server_modified'].strftime('%Y-%m-%d %H:%M')}" if f.get('server_modified') else "")
            for f in files
        ]
        # Auto-select first file (no dropdowns - auto-intent only)
        sel_idx = 0  # Always use first file
        sel = files[0]
        cleansed_paths = [sel["path_lower"]]

        st.caption(f"Selected: {sel['path_lower']}")
        user_q = st.text_area(
            "Your question",
            placeholder="e.g., Why did Aged WIP cost increase in June vs May? Show top plants and drivers.",
            height=100,
            key="qa_question",
        )

        run_btn = st.button("Run Q&A", type="primary", disabled=(not user_q.strip()))
        if run_btn:
            with st.spinner("Thinking with GPT and running tools…"):
                # Phase 6: Monitor Q&A execution
                qa_start_time = time.time()
                if PHASE6_AVAILABLE and monitoring:
                    monitoring.increment("qa.queries", 1, has_files=bool(cleansed_paths))
                    monitoring.gauge("qa.question_length", len(user_q), question_type="user_query")
                
                try:
                    # Import assistant functions only when needed
                    try:
                        from assistant_bridge import run_query_with_files, run_query
                        assistant_available = True
                    except ImportError as e:
                        st.warning(f"Assistant API not available: {e}")
                        assistant_available = False
                    
                    if assistant_available:
                        # Prefer Assistants API with File Search when paths are provided
                        if cleansed_paths:
                            ar = run_query_with_files(user_q, cleansed_paths)
                        else:
                            ar = run_query(user_q)
                        
                        # Phase 6: Monitor Q&A completion
                        if PHASE6_AVAILABLE and monitoring:
                            qa_execution_time = time.time() - qa_start_time
                            monitoring.timing("qa.execution_time", qa_execution_time, has_files=bool(cleansed_paths))
                            monitoring.gauge("qa.answer_length", len(ar.get("answer", "")), question_type="user_query")

                        # Normalize for existing UI
                        answer = ar.get("answer") or ""
                        confidence_obj = ar.get("confidence") or {}
                        confidence_score = confidence_obj.get("score")

                        result = {
                            "final_text": answer,
                            "intent_info": {"intent": "assistant", "confidence": confidence_score},
                            "tool_calls": [],
                            "artifacts": [],
                            "kb_citations": [],
                            "debug": {"thread_id": ar.get("thread_id"), "file_sync_debug": ar.get("debug_file_sync", {})},
                        }
                    else:
                        # Fallback to orchestrator if Assistant not available
                        result = answer_question(
                            user_question=user_q,
                            app_paths=app_paths,
                            cleansed_paths=cleansed_paths,
                            answer_style="concise",
                        )
                except Exception as e:
                    st.error(f"Query processing error: {e}")
                    result = None

            if result:
                st.markdown("### Answer")
                st.write(result.get("final_text", ""))

                # Hidden debug caption (Intent=<auto_routed_intent> | Mode=<resolved_mode>)
                ii = result.get("intent_info", {})
                intent = ii.get('intent', 'unknown')
                mode = 'data_processing'  # This is data processing mode
                st.caption(f"Intent={intent} | Mode={mode}")

                st.markdown("### What I did")
                ii = result.get("intent_info", {})
                st.write(f"- Intent: **{ii.get('intent')}** (confidence {ii.get('confidence')})")
                for call in result.get("tool_calls", []):
                    tool = call.get("tool")
                    meta = call.get("result_meta", {})
                    st.write(f"- Tool: `{tool}` → {json.dumps(meta, ensure_ascii=False)}")

                arts = result.get("artifacts") or []
                if arts:
                    st.markdown("### Artifacts (Dropbox)")
                    for p in arts:
                        st.write(p)

                kb = result.get("kb_citations") or []
                if kb:
                    st.markdown("### KB Sources")
                    for c in kb:
                        st.write(c)

                # Add Sources Drawer and Confidence Badge for Data Processing mode
                st.markdown("---")
                st.markdown("### 📊 Analysis Confidence & Sources")
                
                # Confidence Badge
                confidence_score = ii.get('confidence', 0.5)
                confidence_badge = get_confidence_badge(confidence_score)
                st.markdown(f"**Confidence:** {confidence_badge}", unsafe_allow_html=True)
                
                # Sources Drawer
                sources = result.get("kb_citations", []) or []
                if sources:
                    sources_drawer = SourcesDrawer()
                    sources_drawer.render_inline_sources(sources, confidence_score)
                else:
                    st.info("No sources available for this analysis")

                # Optional: show debug for troubleshooting
                with st.expander("Debug"):
                    st.json(result.get("debug", {}))

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

# =============================================================================
# Required Charts Generation (Phase 1C)
# =============================================================================
st.markdown("---")
st.header("📊 Required Charts Generation")

# Import charting functions
try:
    from charting import (
        create_inventory_aging_waterfall,
        create_usage_vs_stock_scatter,
        create_treemap,
        create_forecast_vs_actual,
        create_moq_histogram
    )
    
    if st.button("🎨 Generate All Required Charts"):
        with st.spinner("Generating required charts..."):
            try:
                # Get sample data for chart generation
                if 'cleaned_sheets' in locals() and cleaned_sheets:
                    # Use the first sheet for chart generation
                    sample_df = list(cleaned_sheets.values())[0]
                    
                    # Generate timestamp for filenames
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    
                    # Generate all 5 required charts
                    charts_generated = []
                    
                    # 1. Inventory aging waterfall
                    aging_path = f"inventory_aging_waterfall_{timestamp}.png"
                    aging_result = create_inventory_aging_waterfall(sample_df, aging_path)
                    if aging_result:
                        charts_generated.append(("Inventory Aging Waterfall", aging_result))
                    
                    # 2. Usage vs stock scatter
                    usage_path = f"usage_vs_stock_scatter_{timestamp}.png"
                    usage_result = create_usage_vs_stock_scatter(sample_df, usage_path)
                    if usage_result:
                        charts_generated.append(("Usage vs Stock Scatter", usage_result))
                    
                    # 3. Treemap
                    treemap_path = f"hierarchical_treemap_{timestamp}.png"
                    treemap_result = create_treemap(sample_df, treemap_path)
                    if treemap_result:
                        charts_generated.append(("Hierarchical Treemap", treemap_result))
                    
                    # 4. Forecast vs actual
                    forecast_path = f"forecast_vs_actual_{timestamp}.png"
                    forecast_result = create_forecast_vs_actual(sample_df, forecast_path)
                    if forecast_result:
                        charts_generated.append(("Forecast vs Actual", forecast_result))
                    
                    # 5. MOQ histogram
                    moq_path = f"moq_histogram_{timestamp}.png"
                    moq_result = create_moq_histogram(sample_df, moq_path)
                    if moq_result:
                        charts_generated.append(("MOQ Histogram", moq_result))
                    
                    # Display results
                    if charts_generated:
                        st.success(f"✅ Generated {len(charts_generated)} charts!")
                        st.markdown("### Generated Charts:")
                        
                        for chart_name, chart_path in charts_generated:
                            st.markdown(f"- **{chart_name}**: `{chart_path}`")
                        
                        # Show chart previews
                        st.markdown("### Chart Previews:")
                        for chart_name, chart_path in charts_generated:
                            st.markdown(f"#### {chart_name}")
                            try:
                                with open(chart_path, 'rb') as f:
                                    st.image(f.read(), caption=chart_name, use_column_width=True)
                            except Exception as e:
                                st.warning(f"Could not display {chart_name}: {e}")
                    else:
                        st.warning("No charts were generated. Check if sample data is available.")
                        
                else:
                    st.info("No cleaned data available. Please run data ingestion first.")
                    
            except Exception as e:
                st.error(f"Chart generation failed: {e}")
                st.exception(e)
    
    else:
        st.info("Click the button above to generate all required charts (inventory aging waterfall, usage-vs-stock scatter, treemap, forecast vs actual, MOQ histogram).")
        
except ImportError as e:
    st.error(f"Charting module not available: {e}")
    st.info("Required charts functionality requires the charting module.")

# =============================================================================
# Phase 6: Production Infrastructure Status
# =============================================================================
if PHASE6_AVAILABLE:
    st.markdown("---")
    st.header("🚀 Phase 6: Production Infrastructure Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Monitoring status
        if monitoring:
            monitoring_health = monitoring.health_check()
            st.metric("Monitoring", monitoring_health["status"])
            st.caption(f"Enabled: {monitoring_health['enabled']}")
        else:
            st.metric("Monitoring", "Not Available")
    
    with col2:
        # Caching status
        if cache:
            cache_health = cache.health_check()
            st.metric("Caching", cache_health["status"])
            cache_stats = cache.get_statistics()
            st.caption(f"Hit Rate: {cache_stats['cache_stats'].get('hit_rate', 0):.1%}")
        else:
            st.metric("Caching", "Not Available")
    
    with col3:
        # Performance metrics
        if monitoring and cache:
            st.metric("Operations", "Active")
            st.caption("Phase 6 Ready")
        else:
            st.metric("Operations", "Limited")
            st.caption("Fallback Mode")
    
    # Cleanup Phase 6 components on app shutdown
    if st.button("🧹 Cleanup Phase 6 Resources"):
        try:
            if cache:
                cache.close()
                st.success("Cache manager closed")
            st.success("Phase 6 cleanup completed")
        except Exception as e:
            st.error(f"Cleanup failed: {e}")

# =============================================================================
# App Cleanup (Streamlit lifecycle)
# =============================================================================
def cleanup_on_exit():
    """Cleanup function for Streamlit app shutdown"""
    if PHASE6_AVAILABLE:
        try:
            if 'cache' in globals() and cache:
                cache.close()
            if 'monitoring' in globals() and monitoring:
                # Send final metrics
                monitoring.increment("app.shutdowns", 1, environment="streamlit")
        except Exception as e:
            pass  # Silent cleanup on exit

# Register cleanup
import atexit
atexit.register(cleanup_on_exit)

# =============================================================================
# Phase 2: Multi-File Comparison Engine
# =============================================================================
st.markdown("---")
st.header("🔄 Multi-File Comparison Engine")

# Import comparison functions
try:
    from phase3_comparison.comparison_utils import compare_wip_aging, compare_inventory, compare_financials
    from path_utils import join_root
    import pandas as pd
    
    # Auto-file pairing detection
    st.subheader("📁 File Pairing & Comparison")
    

    # Get available files for comparison from session state
    comparison_files = []
    
    # Check if we have cleaned sheets in session state
    if 'cleaned_sheets' in st.session_state and st.session_state.cleaned_sheets:
        cleaned_sheets = st.session_state.cleaned_sheets
        for file_path, df in cleaned_sheets.items():
            # Extract period information from filename or data
            period_info = "Unknown"
            if 'q1' in file_path.lower() or 'quarter1' in file_path.lower():
                period_info = "Q1"
            elif 'q2' in file_path.lower() or 'quarter2' in file_path.lower():
                period_info = "Q2"
            elif 'q3' in file_path.lower() or 'quarter3' in file_path.lower():
                period_info = "Q3"
            elif 'q4' in file_path.lower() or 'quarter4' in file_path.lower():
                period_info = "Q4"
            elif 'jan' in file_path.lower() or 'feb' in file_path.lower() or 'mar' in file_path.lower():
                period_info = "Q1"
            elif 'apr' in file_path.lower() or 'may' in file_path.lower() or 'jun' in file_path.lower():
                period_info = "Q2"
            elif 'jul' in file_path.lower() or 'aug' in file_path.lower() or 'sep' in file_path.lower():
                period_info = "Q3"
            elif 'oct' in file_path.lower() or 'nov' in file_path.lower() or 'dec' in file_path.lower():
                period_info = "Q4"
            
            comparison_files.append({
                'path': file_path,
                'period': period_info,
                'rows': len(df),
                'columns': len(df.columns),
                'dataframe': df
            })
    
    # Check for uploaded files if no session state files
    if len(comparison_files) == 0:
        # Provide alternative: direct file upload for comparison
        st.markdown("### 📤 Upload Files for Comparison")
        st.write("You can upload files directly for comparison without running the full ingestion pipeline.")
        
        uploaded_files = st.file_uploader(
            "Choose 2 or more Excel files for comparison",
            type=['xlsx'],
            accept_multiple_files=True,
            help="Select multiple .xlsx files to compare"
        )
        
        if uploaded_files and len(uploaded_files) >= 2:
            st.success(f"✅ Uploaded {len(uploaded_files)} files for comparison")
            
            # Show sheet selection for all files first
            st.markdown("### 📋 Sheet Selection")
            sheet_selections = {}
            
            for uploaded_file in uploaded_files:
                try:
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    
                    if len(sheet_names) > 1:
                        st.info(f"📋 **{uploaded_file.name}** has {len(sheet_names)} sheets: {', '.join(sheet_names)}")
                        selected_sheet = st.selectbox(
                            f"Select sheet from {uploaded_file.name}",
                            sheet_names,
                            key=f"sheet_{uploaded_file.name}_{hash(uploaded_file.name)}"
                        )
                        sheet_selections[uploaded_file.name] = selected_sheet
                    else:
                        sheet_selections[uploaded_file.name] = sheet_names[0]
                except Exception as e:
                    st.error(f"Error reading {uploaded_file.name}: {e}")
            
            # Process files with selected sheets
            st.markdown("### 📊 Processing Selected Sheets")
            
            for uploaded_file in uploaded_files:
                try:
                    selected_sheet = sheet_selections.get(uploaded_file.name, 0)
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    
                    st.success(f"✅ Loaded {uploaded_file.name} - Sheet: {selected_sheet} ({len(df)} rows, {len(df.columns)} cols)")
                    
                    # Extract period information from filename
                    filename = uploaded_file.name.lower()
                    period_info = "Unknown"
                    
                    if 'q1' in filename or 'quarter1' in filename:
                        period_info = "Q1"
                    elif 'q2' in filename or 'quarter2' in filename:
                        period_info = "Q2"
                    elif 'q3' in filename or 'quarter3' in filename:
                        period_info = "Q3"
                    elif 'q4' in filename or 'quarter4' in filename:
                        period_info = "Q4"
                    elif 'jan' in filename or 'january' in filename:
                        period_info = "Q1"
                    elif 'feb' in filename or 'february' in filename:
                        period_info = "Q1"
                    elif 'mar' in filename or 'march' in filename:
                        period_info = "Q1"
                    elif 'apr' in filename or 'april' in filename:
                        period_info = "Q2"
                    elif 'may' in filename:
                        period_info = "Q2"
                    elif 'jun' in filename or 'june' in filename:
                        period_info = "Q2"
                    elif 'jul' in filename or 'july' in filename:
                        period_info = "Q3"
                    elif 'aug' in filename or 'august' in filename:
                        period_info = "Q3"
                    elif 'sep' in filename or 'september' in filename:
                        period_info = "Q3"
                    elif 'oct' in filename or 'october' in filename:
                        period_info = "Q4"
                    elif 'nov' in filename or 'november' in filename:
                        period_info = "Q4"
                    elif 'dec' in filename or 'december' in filename:
                        period_info = "Q4"
                    else:
                        # If no clear period detected, use filename as period
                        period_info = f"File_{len(comparison_files) + 1}"
                    
                    comparison_files.append({
                        'path': uploaded_file.name,
                        'period': period_info,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'dataframe': df
                    })
                except Exception as e:
                    st.error(f"Error reading {uploaded_file.name}: {e}")
    
    # Process comparison files regardless of source (session state or uploaded)
    if len(comparison_files) >= 2:
        st.success(f"✅ Found {len(comparison_files)} files for comparison")
        
        # Display file summary
        st.markdown("### Available Files:")
        for i, file_info in enumerate(comparison_files):
            st.write(f"**{i+1}.** {file_info['period']} - {file_info['path']} ({file_info['rows']} rows, {file_info['columns']} cols)")
        
        # Auto-pairing logic
        st.markdown("### 🔗 Auto-Pairing Strategy")
        
        # Group files by period
        period_groups = {}
        for file_info in comparison_files:
            period = file_info['period']
            if period not in period_groups:
                period_groups[period] = []
            period_groups[period].append(file_info)
        
        # Find pairs for comparison
        comparison_pairs = []
        periods = sorted(period_groups.keys())
        
        for i in range(len(periods) - 1):
            for j in range(i + 1, len(periods)):
                period1, period2 = periods[i], periods[j]
                files1 = period_groups[period1]
                files2 = period_groups[period2]
                
                # Create pairs
                for file1 in files1:
                    for file2 in files2:
                        comparison_pairs.append({
                            'period1': period1,
                            'period2': period2,
                            'file1': file1,
                            'file2': file2,
                            'description': f"{period1} vs {period2}"
                        })
        
        if comparison_pairs:
            st.success(f"🔗 Auto-detected {len(comparison_pairs)} comparison pairs")
            
            # Display comparison pairs
            st.markdown("### Comparison Pairs:")
            for i, pair in enumerate(comparison_pairs):
                st.write(f"**{i+1}.** {pair['description']}")
                st.write(f"   - {pair['file1']['path']} → {pair['file2']['path']}")
            
            # Comparison execution
            st.markdown("### 🚀 Execute Comparison")
            
            # Let user select comparison type and pair
            col1, col2 = st.columns(2)
            
            with col1:
                comparison_type = st.selectbox(
                    "Comparison Type",
                    ["Auto-Detect", "WIP Aging", "Inventory", "Financials"],
                    help="Auto-Detect will analyze data structure to determine best comparison method"
                )
            
            with col2:
                if comparison_pairs:
                    selected_pair_idx = st.selectbox(
                        "Select Comparison Pair",
                        range(len(comparison_pairs)),
                        format_func=lambda x: comparison_pairs[x]['description']
                    )
                    selected_pair = comparison_pairs[selected_pair_idx]
                else:
                    selected_pair = None
            
            if st.button("🔄 Run Comparison Analysis") and selected_pair:
                with st.spinner("Running comparison analysis..."):
                    try:
                        # Prepare dataframes for comparison
                        df1 = selected_pair['file1']['dataframe'].copy()
                        df2 = selected_pair['file2']['dataframe'].copy()
                        
                        # Debug: Show column information
                        st.markdown("### 🔍 Column Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{selected_pair['period1']} Columns:**")
                            st.write(list(df1.columns))
                        
                        with col2:
                            st.markdown(f"**{selected_pair['period2']} Columns:**")
                            st.write(list(df2.columns))
                        
                        # Add period and source information
                        df1['period'] = selected_pair['period1']
                        df1['source_file'] = selected_pair['file1']['path']
                        df2['period'] = selected_pair['period2']
                        df2['source_file'] = selected_pair['file2']['path']
                        
                        # Add lineage columns
                        df1['header_row'] = 'auto-detected'
                        df1['sheet_type'] = 'auto-detected'
                        df2['header_row'] = 'auto-detected'
                        df2['sheet_type'] = 'auto-detected'
                        
                        # Determine comparison strategy
                        if comparison_type == "Auto-Detect":
                            # Analyze data structure to determine type
                            columns1 = set(df1.columns)
                            columns2 = set(df2.columns)
                            
                            # Check for WIP/aging indicators
                            wip_indicators = ['job_no', 'job_name', 'aging', 'wip', 'work_in_progress']
                            if any(indicator in ' '.join(columns1).lower() for indicator in wip_indicators):
                                comparison_type = "WIP Aging"
                            # Check for inventory indicators
                            elif any(indicator in ' '.join(columns1).lower() for indicator in ['part_number', 'sku', 'inventory', 'stock']):
                                comparison_type = "Inventory"
                            # Check for financial indicators
                            elif any(indicator in ' '.join(columns1).lower() for indicator in ['gl_account', 'account', 'financial', 'cost']):
                                comparison_type = "Financials"
                            else:
                                comparison_type = "Inventory"  # Default fallback
                        
                        st.info(f"🔍 Using comparison strategy: **{comparison_type}**")
                        
                        # Execute comparison based on type with local output folder
                        import tempfile
                        import os
                        local_output = os.path.join(tempfile.gettempdir(), "scie_ethos_comparisons")
                        try:
                            os.makedirs(local_output, exist_ok=True)
                            st.info(f"📁 Using output directory: `{local_output}`")
                        except Exception as e:
                            st.error(f"Failed to create output directory: {e}")
                            # Fallback to current directory
                            local_output = "./comparisons"
                            os.makedirs(local_output, exist_ok=True)
                            st.info(f"📁 Using fallback directory: `{local_output}`")
                        
                        # Generate timestamp for output files
                        import time
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        
                        # Enhanced column detection for flexible comparison
                        def detect_value_columns(df):
                            """Detect numeric columns that could be used for comparison"""
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            
                            # Remove system columns
                            system_cols = ['period', 'source_file', 'header_row', 'sheet_type']
                            numeric_cols = [col for col in numeric_cols if col not in system_cols]
                            
                            # Prioritize common inventory columns
                            priority_patterns = [
                                'qty', 'quantity', 'amount', 'value', 'cost', 'price', 
                                'balance', 'remaining', 'stock', 'inventory', 'total',
                                'extended', 'unit', 'each'
                            ]
                            
                            prioritized_cols = []
                            other_cols = []
                            
                            for col in numeric_cols:
                                col_lower = col.lower()
                                if any(pattern in col_lower for pattern in priority_patterns):
                                    prioritized_cols.append(col)
                                else:
                                    other_cols.append(col)
                            
                            return prioritized_cols + other_cols
                        
                        # Show detected columns
                        value_cols_df1 = detect_value_columns(df1)
                        value_cols_df2 = detect_value_columns(df2)
                        
                        st.markdown("### 📊 Detected Numeric Columns")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{selected_pair['period1']} Numeric Columns:**")
                            st.write(value_cols_df1 if value_cols_df1 else "No numeric columns found")
                        
                        with col2:
                            st.markdown(f"**{selected_pair['period2']} Numeric Columns:**")
                            st.write(value_cols_df2 if value_cols_df2 else "No numeric columns found")
                        
                        if comparison_type == "WIP Aging":
                            result = compare_wip_aging([df1, df2], output_folder=local_output)
                        elif comparison_type == "Financials":
                            result = compare_financials([df1, df2], output_folder=local_output)
                        else:  # Default to inventory
                            # Try the original function first
                            try:
                                result = compare_inventory([df1, df2], output_folder=local_output)
                            except ValueError as e:
                                if "No quantity or value columns found" in str(e):
                                    st.warning("⚠️ Standard inventory comparison failed. Using flexible comparison...")
                                    
                                    # Create a simplified comparison using detected columns
                                    if value_cols_df1 and value_cols_df2:
                                        # Use the best common numeric column (prioritize Total columns)
                                        common_cols = set(value_cols_df1) & set(value_cols_df2)
                                        if common_cols:
                                            # Prioritize Total/summary columns
                                            priority_cols = ['Total', 'Aging Total', 'total', 'aging_total', 'sum', 'Sum']
                                            primary_col = None
                                            
                                            # First, try to find a priority column
                                            for priority in priority_cols:
                                                if priority in common_cols:
                                                    primary_col = priority
                                                    break
                                            
                                            # If no priority column found, use the first common column
                                            if not primary_col:
                                                primary_col = list(common_cols)[0]
                                            
                                            st.info(f"📊 Using column '{primary_col}' for comparison (from {len(common_cols)} common numeric columns)")
                                            
                                            # Perform detailed analysis
                                            total_q1 = df1[primary_col].sum()
                                            total_q2 = df2[primary_col].sum()
                                            change = total_q2 - total_q1
                                            change_pct = (change / total_q1 * 100) if total_q1 != 0 else 0
                                            
                                            # Find top items by value
                                            top_q1 = df1.nlargest(5, primary_col)[['description', primary_col]] if 'description' in df1.columns else df1.nlargest(5, primary_col)
                                            top_q2 = df2.nlargest(5, primary_col)[['description', primary_col]] if 'description' in df2.columns else df2.nlargest(5, primary_col)
                                            
                                            # Create Excel file for both local download and Dropbox storage
                                            local_file = f"comparison_q1_vs_q2_{timestamp}.xlsx"
                                            
                                            # Also save to Dropbox merged comparisons folder if available
                                            try:
                                                from dbx_utils import upload_bytes
                                                from path_utils import join_root
                                                dropbox_path = join_root("04_Data/05_Merged_Comparisons", f"comparison_q1_vs_q2_{timestamp}.xlsx")
                                                save_to_dropbox = True
                                                st.info(f"📁 Will save to: Local download + Dropbox: {dropbox_path}")
                                            except ImportError:
                                                save_to_dropbox = False
                                                st.info(f"📁 Will save to: Local download only")
                                            
                                            accessible_file = local_file
                                            
                                            with pd.ExcelWriter(accessible_file, engine='openpyxl') as writer:
                                                # Raw data sheets
                                                df1.to_excel(writer, sheet_name=f'{selected_pair["period1"]}_Data', index=False)
                                                df2.to_excel(writer, sheet_name=f'{selected_pair["period2"]}_Data', index=False)
                                                
                                                # Summary analysis
                                                summary_df = pd.DataFrame({
                                                    'Metric': [
                                                        f'Total {primary_col} - {selected_pair["period1"]}',
                                                        f'Total {primary_col} - {selected_pair["period2"]}',
                                                        'Absolute Change',
                                                        'Percentage Change',
                                                        f'Record Count - {selected_pair["period1"]}',
                                                        f'Record Count - {selected_pair["period2"]}'
                                                    ],
                                                    'Value': [
                                                        f"{total_q1:,.2f}",
                                                        f"{total_q2:,.2f}",
                                                        f"{change:,.2f}",
                                                        f"{change_pct:.1f}%",
                                                        len(df1),
                                                        len(df2)
                                                    ]
                                                })
                                                summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
                                                
                                                # Top items comparison
                                                if not top_q1.empty and not top_q2.empty:
                                                    top_q1.to_excel(writer, sheet_name=f'Top_{selected_pair["period1"]}', index=False)
                                                    top_q2.to_excel(writer, sheet_name=f'Top_{selected_pair["period2"]}', index=False)
                                            
                                            # Upload to Dropbox if available
                                            if save_to_dropbox:
                                                try:
                                                    with open(accessible_file, 'rb') as f:
                                                        file_bytes = f.read()
                                                    upload_bytes(dropbox_path, file_bytes)
                                                    st.success(f"✅ Saved to Dropbox: {dropbox_path}")
                                                except Exception as e:
                                                    st.warning(f"⚠️ Dropbox upload failed: {e}")
                                                    st.info("File is still available for local download")
                                            
                                            # Enhanced AI-like analysis
                                            # Analyze aging buckets for deeper insights
                                            aging_cols = ['0-30 Days', '31-60 Days', '61-90 Days', '91-120 Days', '121-150 Days', '151-180 Days', 'Over 180 Days']
                                            aging_analysis = ""
                                            
                                            # Check which aging columns exist in both datasets
                                            common_aging = [col for col in aging_cols if col in df1.columns and col in df2.columns]
                                            
                                            if common_aging:
                                                aging_changes = {}
                                                for col in common_aging:
                                                    q1_aging = df1[col].sum()
                                                    q2_aging = df2[col].sum()
                                                    aging_changes[col] = q2_aging - q1_aging
                                                
                                                # Find biggest changes
                                                biggest_increase = max(aging_changes.items(), key=lambda x: x[1])
                                                biggest_decrease = min(aging_changes.items(), key=lambda x: x[1])
                                                
                                                if biggest_increase[1] > 0:
                                                    aging_analysis += f"🔴 **Aging Alert**: {biggest_increase[0]} bucket increased by {biggest_increase[1]:,.0f}, indicating potential collection issues.\n\n"
                                                
                                                if biggest_decrease[1] < 0:
                                                    aging_analysis += f"🟢 **Aging Improvement**: {biggest_decrease[0]} bucket decreased by {abs(biggest_decrease[1]):,.0f}, showing better collections.\n\n"
                                            
                                            # Analyze job status if available
                                            status_analysis = ""
                                            if 'Job Status' in df1.columns and 'Job Status' in df2.columns:
                                                q1_statuses = df1['Job Status'].value_counts()
                                                q2_statuses = df2['Job Status'].value_counts()
                                                
                                                # Find status changes
                                                all_statuses = set(q1_statuses.index) | set(q2_statuses.index)
                                                status_changes = {}
                                                for status in all_statuses:
                                                    q1_count = q1_statuses.get(status, 0)
                                                    q2_count = q2_statuses.get(status, 0)
                                                    status_changes[status] = q2_count - q1_count
                                                
                                                # Report significant changes
                                                for status, change in status_changes.items():
                                                    if abs(change) >= 2:  # Only report changes of 2+ jobs
                                                        if change > 0:
                                                            status_analysis += f"📊 **Status Trend**: {change} more jobs in '{status}' status.\n"
                                                        else:
                                                            status_analysis += f"📊 **Status Trend**: {abs(change)} fewer jobs in '{status}' status.\n"
                                                
                                                if status_analysis:
                                                    status_analysis += "\n"
                                            
                                            # Create comprehensive analysis text
                                            analysis_text = f"""
## 🎯 AI-Powered WIP & Inventory Analysis: {selected_pair['period1']} vs {selected_pair['period2']}

### Executive Summary
- **Primary Metric**: {primary_col}
- **{selected_pair['period1']} Total**: ${total_q1:,.0f}
- **{selected_pair['period2']} Total**: ${total_q2:,.0f}
- **Net Change**: ${change:+,.0f} ({change_pct:+.1f}%)
- **Portfolio Size**: {len(df1)} → {len(df2)} jobs

### 🔍 AI-Generated Insights

"""
                                            
                                            # Smart trend analysis
                                            if abs(change_pct) > 20:
                                                analysis_text += f"🚨 **Significant Change Alert**: The {change_pct:+.1f}% change in {primary_col} represents a major shift in your WIP portfolio that requires immediate attention.\n\n"
                                            elif abs(change_pct) > 10:
                                                analysis_text += f"⚠️ **Notable Change**: The {change_pct:+.1f}% change in {primary_col} indicates meaningful portfolio movement worth investigating.\n\n"
                                            elif abs(change_pct) < 2:
                                                analysis_text += f"✅ **Stable Portfolio**: The {change_pct:+.1f}% change shows your WIP portfolio remained relatively stable between periods.\n\n"
                                            
                                            # Portfolio size analysis
                                            job_change = len(df2) - len(df1)
                                            if job_change > 0:
                                                analysis_text += f"📈 **Portfolio Expansion**: Added {job_change} new jobs to your WIP portfolio, indicating business growth or project intake acceleration.\n\n"
                                            elif job_change < 0:
                                                analysis_text += f"📉 **Portfolio Contraction**: Completed or removed {abs(job_change)} jobs from WIP, suggesting improved project velocity or reduced intake.\n\n"
                                            
                                            # Add aging and status insights
                                            analysis_text += aging_analysis
                                            analysis_text += status_analysis
                                            
                                            # Strategic recommendations
                                            analysis_text += "### 💡 Strategic Recommendations\n\n"
                                            
                                            if change_pct > 15:
                                                analysis_text += "🎯 **Action Required**: Consider investigating the drivers of this WIP increase - is it due to new project intake, slower completion rates, or collection delays?\n\n"
                                            elif change_pct < -15:
                                                analysis_text += "🎯 **Positive Trend**: The WIP reduction suggests improved project completion or collection efficiency. Consider analyzing successful practices for replication.\n\n"
                                            
                                            if 'Over 180 Days' in common_aging and aging_changes.get('Over 180 Days', 0) > 0:
                                                analysis_text += "🔴 **Collection Priority**: Focus on the Over 180 Days bucket as it represents the highest risk for bad debt.\n\n"
                                            
                                            analysis_text += "📊 **Next Steps**: Review the detailed Excel workbook for item-level analysis and consider setting up automated alerts for significant WIP changes.\n"
                                            
                                            # Create a simple comparison result
                                            result = {
                                                'metadata': {
                                                    'comparison_type': 'Flexible Inventory',
                                                    'periods': [selected_pair['period1'], selected_pair['period2']],
                                                    'primary_column': primary_col,
                                                    'records_compared': len(df1) + len(df2)
                                                },
                                                'excel_file': accessible_file,
                                                'summary': f"Comparison based on {primary_col} column",
                                                'analysis_text': analysis_text,
                                                'total_delta': change
                                            }
                                            
                                            st.success("✅ Flexible comparison completed!")
                                            
                                            # Store results in session state for Q&A
                                            st.session_state.comparison_result = result
                                            st.session_state.comparison_df1 = df1
                                            st.session_state.comparison_df2 = df2
                                        else:
                                            st.error("❌ **No Common Numeric Columns Found**")
                                            st.info("The selected sheets have completely different column structures. Please:")
                                            st.write("1. **Check if you selected the right sheets** - both files should have similar data types")
                                            st.write("2. **Try different sheet combinations** - use the sheet selectors above")
                                            st.write("3. **Verify data structure** - both sheets should be inventory/WIP data")
                                            
                                            # Show column comparison for debugging
                                            st.markdown("### 🔍 Column Structure Comparison")
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.markdown(f"**{selected_pair['period1']} Columns:**")
                                                for i, col in enumerate(value_cols_df1):
                                                    st.write(f"{i+1}. {col}")
                                            
                                            with col2:
                                                st.markdown(f"**{selected_pair['period2']} Columns:**")
                                                for i, col in enumerate(value_cols_df2):
                                                    st.write(f"{i+1}. {col}")
                                            
                                            st.warning("💡 **Tip**: For WIP aging data, both sheets should have aging buckets like '0-30 Days', '31-60 Days', etc. For inventory data, both should have columns like 'Quantity', 'Value', 'Cost', etc.")
                                            
                                            # Don't proceed with further processing
                                            pass
                                    else:
                                        raise ValueError("No numeric columns found in either file")
                                else:
                                    raise e
                        
                        if result and not isinstance(result, dict):
                            st.error(f"Comparison failed: {result}")
                        elif result:
                            st.success("✅ Comparison completed successfully!")
                            
                            # Display detailed analysis if available
                            if 'analysis_text' in result:
                                st.markdown("### 📊 Detailed Analysis")
                                st.markdown(result['analysis_text'])
                            
                            # Display results
                            st.markdown("### 📊 Comparison Results")
                            
                            # Show metadata
                            if 'metadata' in result:
                                meta = result['metadata']
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Type", meta.get('comparison_type', 'Unknown'))
                                with col2:
                                    st.metric("Periods", f"{meta.get('periods', ['Unknown'])[0]} → {meta.get('periods', ['Unknown'])[-1]}")
                                with col3:
                                    st.metric("Records", meta.get('records_compared', 0))
                            
                            # Show file paths
                            if 'excel_file' in result:
                                st.markdown("### 📁 Generated Files")
                                st.write(f"**Excel Workbook:** `{result['excel_file']}`")
                                st.info("💡 **File Location**: The Excel file is saved in your current working directory and can be downloaded from your browser.")
                                
                                # Add download button with key to prevent UI clearing
                                try:
                                    if os.path.exists(result['excel_file']):
                                        with open(result['excel_file'], 'rb') as file:
                                            file_data = file.read()
                                        
                                        st.download_button(
                                            label="📥 Download Comparison Excel File",
                                            data=file_data,
                                            file_name=os.path.basename(result['excel_file']),
                                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                            key="download_comparison_excel"
                                        )
                                    else:
                                        st.error(f"File not found: {result['excel_file']}")
                                except Exception as e:
                                    st.warning(f"Download not available: {e}")
                                    st.write(f"Debug - Excel file path: {result.get('excel_file', 'No file path')}")
                                
                                if 'json_file' in result:
                                    st.write(f"**JSON Data:** `{result['json_file']}`")
                                if 'summary_file' in result:
                                    st.write(f"**Summary:** `{result['summary_file']}`")
                            
                            # Show delta information if available
                            if 'total_delta' in result:
                                delta = result['total_delta']
                                if delta > 0:
                                    st.success(f"📈 **Total Change:** +{delta:,.2f}")
                                elif delta < 0:
                                    st.error(f"📉 **Total Change:** {delta:,.2f}")
                                else:
                                    st.info(f"➖ **Total Change:** {delta:,.2f} (no change)")
                            
                            # Add Q&A Section
                            st.markdown("### 💬 Ask Questions About This Comparison")
                            
                            # Use form to prevent UI from clearing
                            with st.form("qa_form"):
                                user_question = st.text_input(
                                    "What would you like to know about this comparison?",
                                    placeholder="e.g., What are the top 3 jobs that changed the most? Which aging buckets saw the biggest increase?",
                                    key="comparison_question_form"
                                )
                                
                                analyze_button = st.form_submit_button("🔍 Analyze Question")
                            
                            if analyze_button and user_question:
                                with st.spinner("Analyzing your question..."):
                                    try:
                                        # Enhanced Q&A based on the comparison data
                                        if 'comparison_df1' in st.session_state and 'comparison_df2' in st.session_state:
                                            df1_qa = st.session_state.comparison_df1
                                            df2_qa = st.session_state.comparison_df2
                                            primary_col = result['metadata']['primary_column']
                                            
                                            # Analyze the question for key terms
                                            question_lower = user_question.lower()
                                            
                                            answer = f"**Your Question:** {user_question}\n\n"
                                            
                                            if any(word in question_lower for word in ['top', 'largest', 'biggest', 'highest']):
                                                # Find top changes
                                                if 'Job Name' in df1_qa.columns and 'Job Name' in df2_qa.columns:
                                                    # Merge dataframes to find changes
                                                    merged = pd.merge(df1_qa[['Job Name', primary_col]], 
                                                                    df2_qa[['Job Name', primary_col]], 
                                                                    on='Job Name', how='outer', suffixes=('_Q1', '_Q2'))
                                                    merged = merged.fillna(0)
                                                    merged['Change'] = merged[f'{primary_col}_Q2'] - merged[f'{primary_col}_Q1']
                                                    
                                                    # Get top 3 increases and decreases
                                                    top_increases = merged.nlargest(3, 'Change')
                                                    top_decreases = merged.nsmallest(3, 'Change')
                                                    
                                                    answer += "**📈 Top 3 Increases:**\n"
                                                    for _, row in top_increases.iterrows():
                                                        if row['Change'] > 0:
                                                            answer += f"- {row['Job Name']}: +${row['Change']:,.0f}\n"
                                                    
                                                    answer += "\n**📉 Top 3 Decreases:**\n"
                                                    for _, row in top_decreases.iterrows():
                                                        if row['Change'] < 0:
                                                            answer += f"- {row['Job Name']}: ${row['Change']:,.0f}\n"
                                                else:
                                                    answer += f"Job-level comparison not available. Overall change: ${result['total_delta']:+,.0f}\n"
                                            
                                            elif any(word in question_lower for word in ['aging', 'bucket', 'days']):
                                                # Analyze aging buckets
                                                aging_cols = ['0-30 Days', '31-60 Days', '61-90 Days', '91-120 Days', '121-150 Days', '151-180 Days', 'Over 180 Days']
                                                aging_changes = {}
                                                
                                                for col in aging_cols:
                                                    if col in df1_qa.columns and col in df2_qa.columns:
                                                        q1_total = df1_qa[col].sum()
                                                        q2_total = df2_qa[col].sum()
                                                        aging_changes[col] = q2_total - q1_total
                                                
                                                if aging_changes:
                                                    answer += "**📊 Aging Bucket Changes:**\n"
                                                    for bucket, change in sorted(aging_changes.items(), key=lambda x: abs(x[1]), reverse=True):
                                                        if abs(change) > 1000:  # Only show significant changes
                                                            answer += f"- {bucket}: ${change:+,.0f}\n"
                                                else:
                                                    answer += "Aging bucket analysis not available with current data structure.\n"
                                            
                                            elif any(word in question_lower for word in ['status', 'job status']):
                                                # Analyze job status changes
                                                if 'Job Status' in df1_qa.columns and 'Job Status' in df2_qa.columns:
                                                    q1_status = df1_qa['Job Status'].value_counts()
                                                    q2_status = df2_qa['Job Status'].value_counts()
                                                    
                                                    answer += "**📊 Job Status Changes:**\n"
                                                    all_statuses = set(q1_status.index) | set(q2_status.index)
                                                    for status in all_statuses:
                                                        q1_count = q1_status.get(status, 0)
                                                        q2_count = q2_status.get(status, 0)
                                                        change = q2_count - q1_count
                                                        if change != 0:
                                                            answer += f"- {status}: {q1_count} → {q2_count} ({change:+})\n"
                                                else:
                                                    answer += "Job status analysis not available with current data structure.\n"
                                            
                                            else:
                                                # General analysis
                                                answer += f"""**General Analysis:**

Based on the comparison using '{primary_col}':
- **{selected_pair['period1']} Total**: ${df1_qa[primary_col].sum():,.0f}
- **{selected_pair['period2']} Total**: ${df2_qa[primary_col].sum():,.0f}
- **Net Change**: ${result['total_delta']:+,.0f}
- **Job Count**: {len(df1_qa)} → {len(df2_qa)}

For more specific insights, try asking about:
- "What are the top 3 jobs that changed the most?"
- "Which aging buckets saw the biggest changes?"
- "How did job statuses change between periods?"
"""
                                            
                                            answer += f"\n💡 **Tip**: Check the Excel file '{os.path.basename(result['excel_file'])}' for detailed item-level analysis."
                                            
                                        else:
                                            answer = "Comparison data not available for analysis. Please run the comparison first."
                                        
                                        st.markdown(answer)
                                        
                                    except Exception as e:
                                        st.error(f"Error analyzing question: {e}")
                                        st.write("Debug info:", str(e))
                            
                            # Debug caption showing comparison details
                            st.caption(f"Intent=compare | Mode=data_processing | Strategy={comparison_type} | Pair={selected_pair['period1']}→{selected_pair['period2']}")
                            
                            # Generate comparison charts
                            st.markdown("### 📊 Comparison Charts Generation")
                            
                            try:
                                from charting import create_delta_waterfall, create_aging_shift_chart, create_movers_scatter
                                
                                if st.button("🎨 Generate Comparison Charts", key="generate_charts"):
                                    with st.spinner("Generating comparison charts..."):
                                        try:
                                            charts_generated = []
                                            
                                            # Generate timestamp for chart filenames
                                            chart_timestamp = time.strftime("%Y%m%d_%H%M%S")
                                            
                                            # Use the stored comparison dataframes
                                            if 'comparison_df1' in st.session_state and 'comparison_df2' in st.session_state:
                                                df1_chart = st.session_state.comparison_df1
                                                df2_chart = st.session_state.comparison_df2
                                                
                                                st.info("📊 Generating charts using comparison data...")
                                                
                                                # 1. Delta Waterfall Chart
                                                try:
                                                    waterfall_path = f"delta_waterfall_{chart_timestamp}.png"
                                                    waterfall_result = create_delta_waterfall(df1_chart, waterfall_path)
                                                    if waterfall_result:
                                                        charts_generated.append(("Delta Waterfall", waterfall_result))
                                                except Exception as e:
                                                    st.warning(f"Delta waterfall chart failed: {e}")
                                                
                                                # 2. Aging Shift Chart
                                                try:
                                                    aging_shift_path = f"aging_shift_{chart_timestamp}.png"
                                                    aging_shift_result = create_aging_shift_chart(df1_chart, aging_shift_path)
                                                    if aging_shift_result:
                                                        charts_generated.append(("Aging Shift", aging_shift_result))
                                                except Exception as e:
                                                    st.warning(f"Aging shift chart failed: {e}")
                                                
                                                # 3. Movers Scatter Plot
                                                try:
                                                    movers_path = f"movers_scatter_{chart_timestamp}.png"
                                                    movers_result = create_movers_scatter(df1_chart, movers_path)
                                                    if movers_result:
                                                        charts_generated.append(("Movers Scatter", movers_result))
                                                except Exception as e:
                                                    st.warning(f"Movers scatter chart failed: {e}")
                                                
                                                # Display results
                                                if charts_generated:
                                                    st.success(f"✅ Generated {len(charts_generated)} comparison charts!")
                                                    st.markdown("### Generated Comparison Charts:")
                                                    
                                                    for chart_name, chart_path in charts_generated:
                                                        st.write(f"- **{chart_name}**: `{chart_path}`")
                                                    
                                                    # Show chart previews
                                                    st.markdown("### Chart Previews:")
                                                    for chart_name, chart_path in charts_generated:
                                                        st.markdown(f"#### {chart_name}")
                                                        try:
                                                            st.image(chart_path, caption=chart_name, use_column_width=True)
                                                        except Exception as e:
                                                            st.warning(f"Could not display {chart_name}: {e}")
                                                else:
                                                    st.warning("No comparison charts were generated. This may be due to data format or missing required columns.")
                                            else:
                                                st.error("Comparison data not available. Please run the comparison first.")
                                                
                                        except Exception as e:
                                            st.error(f"Chart generation failed: {e}")
                                            st.info("Charts may require specific data formats or columns that aren't present in your data.")
                                            
                            except ImportError as e:
                                st.error(f"Comparison charting module not available: {e}")
                                st.info("Comparison charts require the enhanced charting module.")
                                
                                # Provide alternative: basic chart using matplotlib
                                if st.button("📊 Generate Basic Charts", key="basic_charts"):
                                    try:
                                        import matplotlib.pyplot as plt
                                        
                                        if 'comparison_result' in st.session_state:
                                            result = st.session_state.comparison_result
                                            df1 = st.session_state.comparison_df1
                                            df2 = st.session_state.comparison_df2
                                            primary_col = result['metadata']['primary_column']
                                            
                                            # Simple bar chart comparison
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            
                                            periods = [selected_pair['period1'], selected_pair['period2']]
                                            totals = [df1[primary_col].sum(), df2[primary_col].sum()]
                                            
                                            bars = ax.bar(periods, totals, color=['#1f77b4', '#ff7f0e'])
                                            ax.set_title(f'Total {primary_col} Comparison')
                                            ax.set_ylabel(primary_col)
                                            
                                            # Add value labels on bars
                                            for bar, total in zip(bars, totals):
                                                height = bar.get_height()
                                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                                       f'{total:,.0f}', ha='center', va='bottom')
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            st.success("✅ Basic comparison chart generated!")
                                        else:
                                            st.error("No comparison data available for charting.")
                                    except Exception as e:
                                        st.error(f"Basic chart generation failed: {e}")
                            
                        else:
                            st.error("Comparison failed - no result returned")
                            
                    except Exception as e:
                        st.error(f"Comparison analysis failed: {e}")
                        st.exception(e)
                        
        else:
            st.warning("⚠️ No comparison pairs found. Need at least 2 files with different periods.")
            
    elif len(comparison_files) == 1:
        st.info("📁 Found 1 file. Need at least 2 files for comparison.")
    else:
        st.info("📁 No files available for comparison. Please run data ingestion first or use the upload option above.")
        
except ImportError as e:
    st.error(f"Comparison module not available: {e}")
    st.info("Multi-file comparison requires the phase3_comparison module.")

# =============================================================================
# Required Charts Generation (Phase 1C)
# =============================================================================
