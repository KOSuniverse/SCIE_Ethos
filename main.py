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
from confidence import get_confidence_badge

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
st.set_page_config(
    page_title="SCIE Ethos LLM", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

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
    # Import and render chat assistant inline
    from chat_ui import render_chat_assistant
    render_chat_assistant()
    st.stop()  # Stop execution here - don't run DP code

# Data Processing mode
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

        run_btn = st.button("Run Analysis", type="primary", disabled=(not user_q.strip()))
        if run_btn:
            with st.spinner("Processing with Enterprise Data Orchestrator…"):
                # Phase 6: Monitor DP execution
                qa_start_time = time.time()
                if PHASE6_AVAILABLE and monitoring:
                    monitoring.increment("dp.queries", 1, has_files=bool(cleansed_paths))
                    monitoring.gauge("dp.question_length", len(user_q), question_type="dp_query")
                
                try:
                    # ENHANCED: Use DP Orchestrator with full enterprise parity
                    from dp_orchestrator import dp_orchestrator
                    
                    # Process query through enhanced orchestrator
                    dp_result = dp_orchestrator.process_dp_query(user_q, st.session_state)
                    
                    # Phase 6: Monitor DP completion
                    if PHASE6_AVAILABLE and monitoring:
                        qa_execution_time = time.time() - qa_start_time
                        monitoring.timing("dp.execution_time", qa_execution_time, has_files=bool(cleansed_paths))
                        monitoring.gauge("dp.answer_length", len(str(dp_result)), question_type="dp_query")

                    # Handle clarification requests
                    if dp_result.get("needs_clarification"):
                        st.warning("🤔 **Clarification Needed**")
                        st.write(dp_result.get("clarification_message", "Multiple options found."))
                        
                        clarification_options = dp_result.get("clarification_options", [])
                        if clarification_options:
                            st.write("**Available options:**")
                            for i, option in enumerate(clarification_options):
                                st.write(f"{i+1}. {option['name']} (Score: {option.get('ranking_score', 0)})")
                        
                        # Store clarification context for follow-up
                        st.session_state["pending_clarification"] = {
                            "original_query": user_q,
                            "options": clarification_options,
                            "clarification_type": dp_result.get("clarification_type", "unknown")
                        }
                        
                        result = None  # Don't show full results yet
                    
                    elif dp_result.get("error"):
                        st.error(f"❌ Analysis failed: {dp_result['error']}")
                        if "suggestions" in dp_result:
                            st.info(f"💡 Suggestion: {dp_result['suggestions']}")
                        result = None
                    
                    else:
                        # ENHANCED: Format using standard_report schema
                        result = {
                            "final_text": dp_result.get("title", "Analysis Complete"),
                            "intent_info": {
                                "intent": dp_result.get("intent", "eda"),
                                "confidence": dp_result.get("confidence", {}).get("score", 0.7)
                            },
                            "dp_result": dp_result,  # Full DP result for enhanced display
                            "tool_calls": [],
                            "artifacts": dp_result.get("artifacts", []),
                            "kb_citations": dp_result.get("citations", []),
                            "debug": {"intent": dp_result.get("intent"), "confidence": dp_result.get("confidence")},
                        }
                        
                        # Clear any pending clarification since we got a result
                        if "pending_clarification" in st.session_state:
                            del st.session_state["pending_clarification"]
                
                except Exception as e:
                    st.error(f"❌ Enhanced DP processing error: {e}")
                    # Fallback to basic orchestrator
                    try:
                        from orchestrator import answer_question
                        result = answer_question(
                            user_question=user_q,
                            app_paths=app_paths,
                            cleansed_paths=cleansed_paths,
                            answer_style="concise",
                        )
                    except Exception as fallback_error:
                        st.error(f"❌ Fallback orchestrator error: {fallback_error}")
                        result = None

            if result:
                # ENHANCED: Display using standard_report schema
                dp_result = result.get("dp_result")
                if dp_result:
                    # Title
                    st.markdown(f"## {dp_result.get('title', 'Analysis Results')}")
                    
                    # Executive Insight
                    executive_insight = dp_result.get('executive_insight', '')
                    if executive_insight:
                        st.markdown("### 🎯 Executive Insight")
                        st.info(executive_insight)
                    
                    # Method & Scope
                    method_scope = dp_result.get('method_and_scope', {})
                    if method_scope:
                        st.markdown("### 🔍 Method & Scope")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Files Analyzed", method_scope.get('files_analyzed', 0))
                        with col2:
                            st.metric("Analysis Type", method_scope.get('analysis_type', 'N/A'))
                        with col3:
                            st.metric("Data Sources", len(method_scope.get('data_sources', [])))
                        
                        # Show file summaries
                        data_sources = method_scope.get('data_sources', [])
                        if data_sources:
                            st.markdown("**Data Sources:**")
                            for i, source in enumerate(data_sources[:3], 1):
                                st.write(f"{i}. {source}")
                    
                    # Evidence & Calculations (Tables + Charts)
                    evidence = dp_result.get('evidence_and_calculations', {})
                    if evidence:
                        st.markdown("### 📊 Evidence & Calculations")
                        
                        # Display tables
                        tables = evidence.get('tables', [])
                        for i, table in enumerate(tables):
                            if isinstance(table, dict):
                                st.markdown(f"**Table {i+1}:** {table.get('title', 'Analysis Table')}")
                                if 'data' in table:
                                    st.dataframe(table['data'])
                            else:
                                st.write(f"Table {i+1}: {table}")
                        
                        # Display charts inline
                        charts = evidence.get('charts', [])
                        if charts:
                            st.markdown("**Charts Generated:**")
                            for chart_group in charts:
                                if isinstance(chart_group, list):
                                    # Handle list of chart paths
                                    for chart_path in chart_group:
                                        if isinstance(chart_path, str) and (chart_path.endswith('.png') or chart_path.endswith('.jpg')):
                                            try:
                                                st.image(chart_path, use_container_width=True)
                                            except Exception as e:
                                                st.write(f"📈 Chart: {chart_path} (display failed: {e})")
                                        else:
                                            st.write(f"📈 {chart_path}")
                                elif isinstance(chart_group, str):
                                    if chart_group.endswith('.png') or chart_group.endswith('.jpg'):
                                        try:
                                            st.image(chart_group, use_container_width=True)
                                        except Exception as e:
                                            st.write(f"📈 Chart: {chart_group} (display failed: {e})")
                                    else:
                                        st.write(f"📈 {chart_group}")
                                else:
                                    st.write(f"📈 Chart: {chart_group}")
                        
                        # Display calculations
                        calculations = evidence.get('calculations', [])
                        if calculations:
                            st.markdown("**Key Calculations:**")
                            for calc in calculations:
                                st.write(f"🔢 {calc}")
                    
                    # Root Causes / Drivers
                    drivers = dp_result.get('root_causes_drivers', [])
                    if drivers:
                        st.markdown("### 🔍 Root Causes & Drivers")
                        for i, driver in enumerate(drivers, 1):
                            st.write(f"{i}. {driver}")
                    
                    # Recommendations
                    recommendations = dp_result.get('recommendations', [])
                    if recommendations:
                        st.markdown("### 💡 Recommendations")
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                    
                    # Confidence & Sources
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Confidence & Sources")
                    
                    confidence = dp_result.get('confidence', {})
                    confidence_score = confidence.get('score', 0.5)
                    confidence_level = confidence.get('level', 'Medium')
                    
                    # Enhanced Confidence Badge with R/A/V/C breakdown
                    confidence_badge = get_confidence_badge(confidence_score, confidence.get('ravc_breakdown'))
                    st.markdown(f"**Confidence:** {confidence_badge}", unsafe_allow_html=True)
                    
                    # R/A/V/C Breakdown
                    ravc = confidence.get('ravc_breakdown', {})
                    if ravc:
                        with st.expander("🔍 Confidence Breakdown (R/A/V/C)"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Retrieval (R)", f"{ravc.get('retrieval_strength_R', 0):.2f}")
                            with col2:
                                st.metric("Agreement (A)", f"{ravc.get('agreement_A', 0):.2f}")
                            with col3:
                                st.metric("Validation (V)", f"{ravc.get('validations_V', 0):.2f}")
                            with col4:
                                st.metric("Citations (C)", f"{ravc.get('citation_density_C', 0):.2f}")
                    
                    # Citations & Sources
                    citations = dp_result.get('citations', [])
                    if citations:
                        sources_drawer = SourcesDrawer()
                        sources_drawer.render_inline_sources(citations, confidence_score)
                    else:
                        st.info("No knowledge base sources available for this analysis")
                    
                    # Limitations & Data Needed
                    limitations = dp_result.get('limits_data_needed', [])
                    if limitations:
                        st.markdown("### ⚠️ Limitations & Data Needed")
                        for limitation in limitations:
                            st.warning(limitation)
                    
                    # Artifacts
                    artifacts = dp_result.get('artifacts', [])
                    if artifacts:
                        st.markdown("### 📁 Generated Artifacts")
                        for artifact in artifacts:
                            st.write(f"📄 {artifact}")
                
                else:
                    # Fallback to basic display
                    st.markdown("### Analysis Results")
                    st.write(result.get("final_text", ""))

                # Hidden debug caption (Intent=<auto_routed_intent> | Mode=<resolved_mode>)
                ii = result.get("intent_info", {})
                intent = ii.get('intent', 'unknown')
                mode = 'data_processing_enhanced'  # Enhanced DP mode
                st.caption(f"Intent={intent} | Mode={mode}")

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

# NOTE: Multi-File Comparison Engine is now integrated into the Enhanced Data Processing mode above.
# Users can perform comparisons by asking natural language questions like "Compare R401 vs R402" in the DP chat.



# NOTE: Multi-File Comparison and Charts are now integrated into the Enhanced Data Processing mode above.
# Users can get comparisons and charts by asking questions like "Compare R401 vs R402" in the DP chat.

# =============================================================================
# Phase 5A: Usage Dashboard
# =============================================================================
st.markdown("---")
st.header("📊 Usage Dashboard")
st.markdown("**Phase 5A**: Query logs and usage analytics")

try:
    import sys
    sys.path.append('PY Files')
    from phase5_governance.usage_dashboard import UsageDashboard
    
    dashboard = UsageDashboard()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Full Dashboard", "📋 Summary Widget"])
    
    with tab1:
        dashboard.render_usage_page()
    
    with tab2:
        dashboard.render_usage_summary_widget()
        
        # Show missing data report summary
        st.markdown("---")
        st.subheader("🔍 Data Quality Summary")
        
        try:
            from phase5_governance.data_gap_analyzer import DataGapAnalyzer
            
            gap_analyzer = DataGapAnalyzer()
            
            # Check if missing data report exists
            if gap_analyzer.report_path.exists():
                with open(gap_analyzer.report_path, 'r') as f:
                    import json
                    report = json.load(f)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Analyses", report.get("total_analyses", 0))
                
                with col2:
                    avg_impact = report.get("summary", {}).get("avg_impact_score", 0)
                    st.metric("Avg Impact", f"{avg_impact:.1f}/10")
                
                with col3:
                    top_missing = report.get("top_missing_fields", {}).get("critical", [])
                    st.metric("Top Missing", len(top_missing))
                
                if st.button("📋 View Full Missing Data Report"):
                    st.json(report)
            else:
                st.info("Missing data report will be generated after several queries.")
                
        except ImportError:
            st.warning("⚠️ Phase 5B Data Gap Analyzer not available")
            
except ImportError:
    st.warning("⚠️ Phase 5A Usage Dashboard not available")
    st.info("Usage analytics will be available once the phase5_governance module is properly installed.")
