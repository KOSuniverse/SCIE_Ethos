# PY Files/path_utils.py
from __future__ import annotations
import os
from dataclasses import dataclass

try:
    import streamlit as st
except Exception:
    st = None

def _secret(key: str, default: str = "") -> str:
    if st is not None and hasattr(st, "secrets"):
        v = st.secrets.get(key)
        if v is not None:
            return str(v)
    return os.getenv(key, default)

# Canonical logical folder names (unchanged from your Colab design)
LOGICAL_FOLDERS = [
    "00_Raw_Files",
    "01_Cleansed_Files",
    "02_EDA_Charts",
    "03_Summaries",
    "04_Metadata",
    "05_Merged_Comparisons",  # optional
]

@dataclass
class ProjectPaths:
    dropbox_root: str
    raw_folder: str
    cleansed_folder: str
    charts_folder: str
    summaries_folder: str
    metadata_folder: str
    merged_folder: str
    master_metadata_path: str
    env_name: str
    # S3 mirrors (optional usage)
    s3_bucket: str
    s3_prefix: str
    s3_base: str

def get_project_paths() -> ProjectPaths:
    # Dropbox app is app‑folder scoped, so API paths **do not** include "/Apps/...".
    # You already validated the root path format: "/Project_Root"
    dbx_root = _secret("DROPBOX_ROOT", "/Project_Root")
    base = f"{dbx_root}/04_Data"

    raw = f"{base}/00_Raw_Files"
    cln = f"{base}/01_Cleansed_Files"
    charts = f"{base}/02_EDA_Charts"
    sums = f"{base}/03_Summaries"
    meta = f"{base}/04_Metadata"
    merged = f"{base}/05_Merged_Comparisons"

    env_name = _secret("ENV_NAME", "prod")

    # S3 mirrors (optional)
    s3_bucket = _secret("S3_BUCKET")
    s3_prefix = _secret("S3_PREFIX").rstrip("/")
    s3_base = f"s3://{s3_bucket}/{s3_prefix}/04_Data" if (s3_bucket and s3_prefix) else ""

    return ProjectPaths(
        dropbox_root=dbx_root,
        raw_folder=raw,
        cleansed_folder=cln,
        charts_folder=charts,
        summaries_folder=sums,
        metadata_folder=meta,
        merged_folder=merged,
        master_metadata_path=f"{meta}/master_metadata_index.json",
        env_name=env_name,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_base=s3_base,
    )
