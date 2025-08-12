# file_utils.py

import os
import json
from pathlib import Path

# Import constants and path utilities
try:
    from .constants import PROJECT_ROOT, DATA_ROOT, META_DIR, CLEANSED
    from .path_utils import canon_path, join_root
except ImportError:
    # Fallback for standalone usage
    from constants import PROJECT_ROOT, DATA_ROOT, META_DIR, CLEANSED
    from path_utils import canon_path, join_root

# Default folder structure (can be overridden by configuration)
DEFAULT_FOLDER_STRUCTURE = [
    "04_Data/00_Raw_Files",
    "04_Data/01_Cleansed_Files",
    "04_Data/02_EDA_Charts",
    "04_Data/03_Summaries",
    "04_Data/04_Metadata",
    "04_Data/05_Merged_Comparisons",
    "04_Data/Models",
    "06_LLM_Knowledge_Base"
]

def ensure_folder_structure(root_path: str = PROJECT_ROOT, folder_structure: list = None):
    """
    Create required folder structure with proper path canonicalization.
    
    Args:
        root_path: Root path for folder creation (defaults to PROJECT_ROOT)
        folder_structure: List of relative folder paths to create
    
    Returns:
        dict: Status of folder creation
    """
    # Use canonical root path
    root_path = canon_path(root_path)
    Path(root_path).mkdir(parents=True, exist_ok=True)
    
    # Create standard structure if none provided
    folders = folder_structure or [
        "04_Data/01_Cleansed_Files",
        "04_Data/02_EDA_Charts", 
        "04_Data/03_Summaries",
        "04_Data/04_Metadata",
        "04_Data/05_Merged_Comparisons",
        "06_LLM_Knowledge_Base/chunks"
    ]
    
    created_folders = []
    for rel in folders:
        folder_path = Path(root_path) / rel
        folder_path.mkdir(parents=True, exist_ok=True)
        created_folders.append(str(folder_path))
    
    return {
        "status": "created",
        "root_path": root_path,
        "created_folders": created_folders
    }

def get_metadata_path(root_path: str = PROJECT_ROOT, metadata_subpath: str = "04_Data/04_Metadata/master_metadata_index.json") -> str:
    """
    Returns the canonical path to the master metadata index.
    Uses META_DIR constant for consistency.
    """
    return canon_path(f"{META_DIR}/master_metadata_index.json")

def list_cleaned_files(root_path: str = PROJECT_ROOT, cleansed_subpath: str = "04_Data/01_Cleansed_Files") -> list:
    """
    Returns a list of cleaned Excel file paths using canonical paths.
    Uses CLEANSED constant for consistency.
    """
    folder_path = Path(canon_path(CLEANSED))
    if not folder_path.exists():
        return []
    
    return sorted(str(p) for p in folder_path.glob("*.xlsx") if p.is_file())
