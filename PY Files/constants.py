# constants.py

import os
try:
    import streamlit as st
    _DBX = st.secrets.get("DROPBOX_ROOT", os.getenv("DROPBOX_ROOT", "")).strip("/")
    _NS = st.secrets.get("DROPBOX_NAMESPACE", os.getenv("DROPBOX_NAMESPACE", "")).strip("/")
    print(f"DEBUG constants.py: DROPBOX_ROOT='{_DBX}', DROPBOX_NAMESPACE='{_NS}'")
except Exception as e:
    print(f"DEBUG constants.py: Streamlit import failed: {e}")
    _DBX = os.getenv("DROPBOX_ROOT", "").strip("/")
    _NS = os.getenv("DROPBOX_NAMESPACE", "").strip("/")

# Construct full Dropbox path from namespace and root
if _DBX and _NS:
    PROJECT_ROOT = f"/{_NS}/{_DBX}"
    print(f"DEBUG constants.py: Using namespace+root: {PROJECT_ROOT}")
elif _DBX:
    PROJECT_ROOT = f"/{_DBX}"
    print(f"DEBUG constants.py: Using root only: {PROJECT_ROOT}")
else:
    PROJECT_ROOT = "/Project_Root"  # Default fallback matching Dropbox structure
    print(f"DEBUG constants.py: Using fallback: {PROJECT_ROOT}")

# Canonical subfolders (matching Dropbox structure from screenshot)
DATA_ROOT = f"{PROJECT_ROOT}/04_Data"                    # type: str
KB_ROOT   = f"{PROJECT_ROOT}/06_LLM_Knowledge_Base"      # type: str
META_DIR  = f"{DATA_ROOT}/04_Metadata"                   # type: str
CLEANSED  = f"{DATA_ROOT}/01_Cleansed_Files"             # type: str
EDA_CHART = f"{DATA_ROOT}/02_EDA_Charts"                 # type: str
COMPARES  = f"{DATA_ROOT}/05_Merged_Comparisons"         # type: str

# Local writable workspace for artifacts/logs; safe on server
import os as _os, tempfile as _tmp
LOCAL_WORK_ROOT = _os.getenv("LOCAL_WORK_ROOT", str(_tmp.gettempdir()) + "/ethos")  # type: str
LOCAL_META_DIR  = f"{LOCAL_WORK_ROOT}/metadata"  # type: str

# --- Legacy Directory Constants (for backward compatibility) ---
RAW_FILES_DIR = "04_Data/00_Raw_Files"
CLEANSED_FILES_DIR = "04_Data/01_Cleansed_Files"
EDA_CHARTS_DIR = "04_Data/02_EDA_Charts"
SUMMARY_DIR = "04_Data/03_Summaries"
METADATA_DIR = "04_Data/04_Metadata"
COMPARISON_DIR = "04_Data/05_Merged_Comparisons"
MODELS_DIR = "04_Data/06_Models"
KNOWLEDGE_BASE_DIR = "06_LLM_Knowledge_Base"

# --- Master Files ---
MASTER_METADATA_FILE = f"{METADATA_DIR}/master_metadata_index.json"
GLOBAL_ALIAS_FILE = f"{METADATA_DIR}/global_column_aliases.json"
SESSION_LOG_FILE = f"{METADATA_DIR}/query_log.json"
ERROR_LOG_FILE = f"{METADATA_DIR}/error.log"

# --- Default Column Types ---
DEFAULT_ID_COLUMNS = ["part_no", "job_no", "item_id"]
DEFAULT_DATE_COLUMNS = ["date", "last_used", "received_date"]

# --- Intent Types ---
INTENT_TYPES = [
    "compare", "root_cause", "forecast", "summarize",
    "eda", "rank", "anomaly", "optimize", "filter"
]
