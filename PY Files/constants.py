# constants.py

# --- Root Data Paths ---
RAW_FILES_DIR = "04_Data/00_Raw_Files"
CLEANSED_FILES_DIR = "04_Data/01_Cleansed_Files"
EDA_CHARTS_DIR = "04_Data/02_EDA_Charts"
SUMMARY_DIR = "04_Data/03_Summaries"
METADATA_DIR = "04_Data/04_Metadata"
COMPARISON_DIR = "04_Data/05_Merged_Comparisons"
MODELS_DIR = "04_Data/Models"
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
