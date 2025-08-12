# main.py — DROP-IN

import os
import io
import json
import uuid
import time
import ast
import datetime
import requests
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
        lines.append(f"- Records: {s.get('record_count')}")
        eda = (s.get("eda_text") or "").strip()
        if eda:
            lines.append("")
            lines.append(eda)
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
            - �📊 **File Selection**: Choose specific cleansed files for analysis  
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
    # Auto-detect Dropbox root once
    if "dbx_root" not in st.session_state:
        # tries /Project_Root (App Folder) then /Apps/Ethos LLM/Project_Root (Full)
        def _detect_dropbox_root(list_func):
            candidates = ["/Project_Root", "/Apps/Ethos LLM/Project_Root"]
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

    up = st.file_uploader("Upload an Excel or CSV file (saved to Dropbox RAW, then ingested)", type=["xlsx", "xlsm", "csv"])

    if up is not None:
        try:
            # 1) Read upload bytes
            up.seek(0)
            file_bytes = up.read()
            filename = up.name

            # 2) Save upload to RAW in Dropbox
            raw_dest = f"{app_paths.dbx_raw_folder}/{filename}"
            upload_bytes(raw_dest, file_bytes)
            st.success(f"Saved to Dropbox RAW: {raw_dest}")

            # 3) Run ingest pipeline on the uploaded bytes (not the RAW path)
            cleaned_sheets, meta = run_ingest_pipeline(
                source=file_bytes,
                filename=filename,
                paths=app_paths
            )

            # 4) Show per-sheet heads and metadata/summaries
            st.subheader("Run metadata (Dropbox ingest)")
            st.json(meta)

            st.subheader("Sheets cleaned (preview)")
            st.write(list(cleaned_sheets.keys()))
            for sname, df in cleaned_sheets.items():
                st.markdown(f"### Sheet: `{sname}`")
                st.dataframe(df.head(5), use_container_width=True)

            if "sheets" in meta:
                st.markdown("### Per-sheet summaries")
                rows = []
                for s in meta["sheets"]:
                    rows.append({
                        "sheet_name": s.get("sheet_name"),
                        "normalized_sheet_type": s.get("normalized_sheet_type"),
                        "records": s.get("record_count"),
                        "summary": s.get("summary_text"),
                        "name_hint": s.get("name_implied_type"),
                        "feature_hint": s.get("feature_implied_type"),
                        "resolution": s.get("type_resolution"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # 5) Append metadata to master index (Dropbox)
            meta_path = getattr(app_paths, "dbx_master_metadata_path", None)
            if not meta_path:
                st.error("Dropbox metadata path not set.")
            else:
                try:
                    existing = json.loads(dbx_read_bytes(meta_path).decode("utf-8"))
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
                existing.append(meta)
                upload_json(meta_path, existing)
                st.success(f"✅ Metadata updated at: {meta_path}")

            st.info("Next: use 'Process a RAW workbook (Dropbox ➜ Dropbox)' to produce the Cleansed workbook.")

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
        raw_files = dbx_list_data_files(raw_dbx)  # Now supports both Excel and CSV
    else:
        st.info("Switch to 'Dropbox' mode in the ingest expander to process cloud files.")
except Exception as e:
    st.error(f"Could not list RAW files: {e}")

if raw_files:
    labels = [f'{f["name"]} ({f.get("file_type", "unknown")})  ·  {f["path_lower"]}' for f in raw_files]
    choice = st.selectbox("Pick a RAW file (Excel or CSV)", options=labels, index=0)

    # Quick peek
    if st.button("🔎 Preview RAW file"):
        try:
            sel = raw_files[labels.index(choice)]["path_lower"]
            file_type = raw_files[labels.index(choice)].get("file_type", "excel")
            b = dbx_read_bytes(sel)
            
            if file_type == "csv":
                # Handle CSV file
                import io
                df_preview = pd.read_csv(io.BytesIO(b), nrows=5)
                st.success(f"Loaded CSV: {sel}")
                st.caption("Preview (first 5 rows):")
                st.dataframe(df_preview, use_container_width=True)
            else:
                # Handle Excel file
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
            sel = raw_files[labels.index(choice)]["path_lower"]
            filename = raw_files[labels.index(choice)]["name"]

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
                
                cleaned_sheets, metadata = run_pipeline(b, filename, app_paths)
                
                # Clear heartbeat
                heartbeat_status.empty()
                
                step_status.write("✅ Step 1/5: Data cleansing completed")
                progress_bar.progress(40)

            # 3) Enhanced EDA Generation and Display
            st.subheader("📈 Data Analysis & Insights")
            
            # Import enhanced EDA tools
            try:
                from phase2_analysis.eda_runner import run_gpt_eda_actions, create_enhanced_eda_summary
                from eda import generate_eda_summary
                eda_available = True
            except ImportError:
                st.warning("⚠️ Enhanced EDA modules not available. Using basic analysis.")
                eda_available = False

            all_insights = {}
            all_chart_paths = []
            
            for sheet_name, df in cleaned_sheets.items():
                with st.expander(f"📊 Analysis: {sheet_name} ({len(df)} rows, {len(df.columns)} columns)"):
                    
                    # Basic stats display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        missing_pct = round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
                        st.metric("Missing Data", f"{missing_pct}%")
                    
                    # Show sample data
                    st.write("**Sample Data:**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    if eda_available and len(df) > 0:
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
                                        
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        
                                        # Create histogram
                                        df[col].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
                                        ax.set_title(f'Distribution: {col}', fontsize=14, fontweight='bold')
                                        ax.set_xlabel(col, fontsize=12)
                                        ax.set_ylabel('Frequency', fontsize=12)
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
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        
                                        ax.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, color='coral')
                                        ax.set_title(f'Relationship: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
                                        ax.set_xlabel(x_col, fontsize=12)
                                        ax.set_ylabel(y_col, fontsize=12)
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
                                    
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    
                                    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', 
                                              center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
                                    ax.set_title('Correlation Analysis', fontsize=16, fontweight='bold', pad=20)
                                    
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
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        
                                        top_data.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
                                        ax.set_title(f'Top 10 {group_col} by {metric_col}', fontsize=14, fontweight='bold')
                                        ax.set_xlabel(group_col, fontsize=12)
                                        ax.set_ylabel(metric_col, fontsize=12)
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
                            
                            # Display charts inline if possible
                            st.write("**📊 Generated Visualizations:**")
                            
                            chart_display_container = st.container()
                            with chart_display_container:
                                # Show downloadable links and try to display charts
                                for i, path in enumerate(chart_paths):
                                    chart_name = path.split('/')[-1]
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"📊 {chart_name}")
                                    with col2:
                                        # Create a download link hint
                                        st.caption("💾 Saved to Dropbox")
                                    
                                    # Try to display the chart by reading it back from Dropbox
                                    try:
                                        chart_bytes = dbx_read_bytes(path)
                                        st.image(chart_bytes, caption=chart_name, use_container_width=True)
                                    except Exception as display_e:
                                        st.caption(f"⚠️ Chart saved but display failed: {display_e}")
                                    
                                    # Add some spacing between charts
                                    if i < len(chart_paths) - 1:
                                        st.write("---")
                                        
                        else:
                            st.info("ℹ️ No charts generated (insufficient data or columns)")
                    
                    else:
                        # Basic EDA fallback
                        basic_summary = generate_eda_summary(df) if 'generate_eda_summary' in globals() else {}
                        if basic_summary:
                            st.write("**Basic Statistics:**")
                            if "descriptive_stats" in basic_summary:
                                st.dataframe(pd.DataFrame(basic_summary["descriptive_stats"]).T)

            step_status.write("✅ Step 2/5: Data analysis completed")
            progress_bar.progress(70)

            # 4) Save cleansed workbook back to Dropbox
            step_status.write("🔄 Step 3/5: Saving cleansed workbook...")
            out_bytes = build_xlsx_bytes_from_sheets(cleaned_sheets)
            out_base = filename.rsplit(".xlsx", 1)[0]
            cleansed_dbx = getattr(app_paths, "dbx_cleansed_folder", None)
            if not cleansed_dbx:
                raise RuntimeError("Dropbox cleansed folder not set (switch to 'Dropbox' mode).")
            out_path = f"{cleansed_dbx}/{out_base}_cleansed.xlsx"
            upload_bytes(out_path, out_bytes)
            step_status.write("✅ Step 3/5: Cleansed workbook saved")
            progress_bar.progress(85)

            # 5) Update master metadata on Dropbox
            step_status.write("🔄 Step 4/5: Updating metadata...")
            progress_bar.progress(85)
            
            # Add EDA insights to metadata
            if all_insights:
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
                    step_status.write(f"🔄 Step 4/5: Updating metadata (attempt {attempt + 1}/{max_retries})...")
                    
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
                    step_status.write("✅ Step 4/5: Metadata updated successfully")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        step_status.write(f"⚠️ Metadata update attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)  # Wait before retry
                    else:
                        st.warning(f"⚠️ Metadata update failed after {max_retries} attempts: {e}")
                        st.info("📊 Data processing completed successfully, but metadata sync had network issues.")
            
            progress_bar.progress(90)

            # 6) Save per-file summaries to Dropbox (JSON + Markdown)
            step_status.write("🔄 Step 5/5: Saving summaries...")
            progress_bar.progress(95)
            
            summaries_dbx = getattr(app_paths, "dbx_summaries_folder", None)
            if summaries_dbx:
                summary_saved = False
                
                # Save run metadata with retry
                for attempt in range(max_retries):
                    try:
                        step_status.write(f"🔄 Step 5/5: Saving metadata summary (attempt {attempt + 1}/{max_retries})...")
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
                        step_status.write(f"🔄 Step 5/5: Saving executive summary (attempt {attempt + 1}/{max_retries})...")
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

                step_status.write("✅ Step 5/5: All processing completed!")
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
                step_status.write("✅ Step 5/5: Processing completed (summaries skipped)")
                progress_bar.progress(100)

            # Display final insights summary
            if all_insights:
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

            # Display Executive Summary on screen
            st.subheader("📋 Executive Summary")
            try:
                # Generate and display the executive summary
                summary_markdown = _build_summary_markdown(metadata)
                st.markdown(summary_markdown)
                
                # Also show key insights in a more digestible format
                st.subheader("🔍 Key Insights by Sheet")
                
                if "sheets" in metadata:
                    for sheet_info in metadata["sheets"]:
                        sheet_name = sheet_info.get("sheet_name", "Unknown")
                        sheet_type = sheet_info.get("normalized_sheet_type", "Unknown")
                        record_count = sheet_info.get("record_count", 0)
                        summary_text = sheet_info.get("summary_text", "No summary available")
                        eda_text = sheet_info.get("eda_text", "")
                        
                        with st.expander(f"📊 {sheet_name} ({sheet_type}) - {record_count:,} records"):
                            if summary_text and summary_text != "No summary available":
                                st.write("**Summary:**")
                                st.write(summary_text)
                            
                            if eda_text:
                                st.write("**Analysis:**")
                                st.write(eda_text)
                            
                            # Show supply chain insights if available
                            if sheet_name in all_insights:
                                insights = all_insights[sheet_name]
                                if insights.get("supply_chain_insights"):
                                    st.write("**Supply Chain Metrics:**")
                                    for category, info in insights["supply_chain_insights"].items():
                                        if info.get("total_value", 0) > 0:
                                            st.write(f"- **{category.title()}**: ${info['total_value']:,.2f}")
                                
                                # Show dataset overview
                                if insights.get("dataset_overview"):
                                    overview = insights["dataset_overview"]
                                    st.write("**Data Quality:**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Missing Data", f"{overview.get('missing_data_percentage', 0):.1f}%")
                                    with col2:
                                        chart_count = len(insights.get("charts_generated", []))
                                        st.metric("Charts Generated", chart_count)
                
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
        cln_files = dbx_list_data_files(cln_dbx)  # Now supports both Excel and CSV
    else:
        st.info("Switch to 'Dropbox' mode to browse Cleansed files.")
except Exception as e:
    st.error(f"Could not list Cleansed files: {e}")

if cln_files:
    cln_labels = [f'{f["name"]} ({f.get("file_type", "unknown")})  ·  {f["path_lower"]}' for f in cln_files]
    cln_choice = st.selectbox("Pick a Cleansed file (Excel or CSV)", options=cln_labels, index=0, key="cln_pick")
    if st.button("🔎 Preview Cleansed file"):
        try:
            sel = cln_files[cln_labels.index(cln_choice)]["path_lower"]
            file_type = cln_files[cln_labels.index(cln_choice)].get("file_type", "excel")
            b = dbx_read_bytes(sel)
            
            if file_type == "csv":
                # Handle CSV file
                df_preview = pd.read_csv(io.BytesIO(b), nrows=5)
                st.success(f"Loaded CSV: {sel}")
                st.caption("Preview (first 5 rows):")
                st.dataframe(df_preview, use_container_width=True)
            else:
                # Handle Excel file
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
            res = build_or_update_knowledgebase(
                project_root=PROJECT_ROOT,
                scan_folders=None,
                force_rebuild=force_rebuild,
                include_text_files=include_text
            )
        st.success("KB build complete")
        st.json(res)

# =========================
# 4) Ask questions about a Cleansed workbook (cloud-only)
# =========================
import datetime as _dt
from orchestrator import answer_question
from dbx_utils import list_data_files

st.markdown("## 4) Ask questions about a Cleansed workbook")

cleansed_folder = getattr(app_paths, "dbx_cleansed_folder", None)
if not cleansed_folder:
    st.warning("Cleansed folder not configured on Dropbox. Set app_paths.dbx_cleansed_folder.")
else:
    try:
        files = list_data_files(cleansed_folder)  # Now supports both Excel and CSV
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
        sel_idx = st.selectbox("Choose a cleansed file (Excel or CSV):", range(len(files)), format_func=lambda i: nice_labels[i], key="qa_select_clean")
        sel = files[sel_idx]
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
                try:
                    result = answer_question(
                        user_question=user_q,
                        app_paths=app_paths,
                        cleansed_paths=cleansed_paths,
                        answer_style="concise",
                    )
                except Exception as e:
                    st.error(f"Orchestrator error: {e}")
                    result = None

            if result:
                st.markdown("### Answer")
                st.write(result.get("final_text", ""))

                st.markdown("### What I did")
                ii = result.get("intent_info", {})
                st.write(f"- Intent: **{ii.get('intent')}** (confidence {ii.get('confidence')}) — {ii.get('reason')}")
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
