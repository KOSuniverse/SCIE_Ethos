# PY Files/path_utils.py
from __future__ import annotations
import os
import re
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

def canon_path(p: str) -> str:
    """
    Canonicalize path with proper case normalization for folder names.
    
    Args:
        p: Path string to normalize
        
    Returns:
        str: Canonicalized path with proper case
    """
    if not p:
        return p
    
    # Normalize slashes
    p = p.replace("\\", "/")
    
    # Replace project_root variations (preserve case if no secret is set)
    dropbox_root = _secret("DROPBOX_ROOT", "")
    if dropbox_root:
        p = p.replace("/project_root", f"/{dropbox_root}")
        p = p.replace("/Project_Root", f"/{dropbox_root}")
    # If no secret, preserve the original case to avoid mismatch
    
    # Case-insensitive folder normalization
    p = re.sub(r"/04_data/", "/04_Data/", p, flags=re.I)
    p = re.sub(r"/06_llm_knowledge_base/", "/06_LLM_Knowledge_Base/", p, flags=re.I)
    
    return p

def join_root(*parts: str) -> str:
    """
    Join path parts with PROJECT_ROOT and canonicalize.
    
    Args:
        *parts: Path parts to join
        
    Returns:
        str: Canonicalized full path
    """
    root = _secret("DROPBOX_ROOT", "/Project_Root").strip("/")
    parts_clean = [s.strip("/") for s in parts if s]
    joined = "/" + "/".join([root] + parts_clean)
    return canon_path(joined)

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
    # S3 mirrors (optional)
    s3_bucket: str
    s3_prefix: str
    s3_base: str

def get_project_paths() -> ProjectPaths:
    # Dropbox app is app-folder scoped, so do NOT include "/Apps/...".
    dbx_root = _secret("DROPBOX_ROOT", "/Project_Root")
    base = f"{dbx_root}/04_Data"

    raw     = f"{base}/00_Raw_Files"
    cleansed= f"{base}/01_Cleansed_Files"
    charts  = f"{base}/02_EDA_Charts"
    sums    = f"{base}/03_Summaries"
    meta    = f"{base}/04_Metadata"
    merged  = f"{base}/05_Merged_Comparisons"

    env_name = _secret("ENV_NAME", "prod")

    s3_bucket = _secret("S3_BUCKET")
    s3_prefix = _secret("S3_PREFIX").rstrip("/")
    s3_base = f"s3://{s3_bucket}/{s3_prefix}/04_Data" if (s3_bucket and s3_prefix) else ""

    return ProjectPaths(
        dropbox_root=dbx_root,
        raw_folder=raw,
        cleansed_folder=cleansed,
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

# Export main functions for clean imports
__all__ = ["canon_path", "join_root", "get_project_paths", "ProjectPaths"]

