# file_utils.py

import os
import json

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

def ensure_folder_structure(root_path: str, folder_structure: list = None):
    """
    Creates the required folder structure under root_path if missing.
    Supports custom folder structures for different deployment environments.
    """
    folders = folder_structure or DEFAULT_FOLDER_STRUCTURE

    for folder in folders:
        full_path = os.path.join(root_path, folder)
        os.makedirs(full_path, exist_ok=True)

def get_metadata_path(root_path: str, metadata_subpath: str = "04_Data/04_Metadata/master_metadata_index.json") -> str:
    """
    Returns the full path to the master metadata index.
    Supports custom metadata paths for different deployment environments.
    """
    return os.path.join(root_path, metadata_subpath)

def list_cleaned_files(root_path: str, cleansed_subpath: str = "04_Data/01_Cleansed_Files") -> list:
    """
    Returns a list of cleaned Excel file paths from the expected folder.
    Supports custom cleansed folder paths for different deployment environments.
    """
    folder = os.path.join(root_path, cleansed_subpath)
    if not os.path.exists(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith((".xlsx", ".csv"))  # Support both Excel and CSV
    ]
