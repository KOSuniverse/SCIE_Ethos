# file_utils.py

import os
import json

def ensure_folder_structure(root_path: str):
    """
    Creates the required folder structure under Project_Root if missing.
    """
    folders = [
        "04_Data/00_Raw_Files",
        "04_Data/01_Cleansed_Files",
        "04_Data/02_EDA_Charts",
        "04_Data/03_Summaries",
        "04_Data/04_Metadata",
        "04_Data/05_Merged_Comparisons",
        "04_Data/Models",
        "06_LLM_Knowledge_Base"
    ]

    for folder in folders:
        full_path = os.path.join(root_path, folder)
        os.makedirs(full_path, exist_ok=True)

def get_metadata_path(root_path: str) -> str:
    """
    Returns the full path to the master metadata index.
    """
    return os.path.join(root_path, "04_Data/04_Metadata/master_metadata_index.json")

def list_cleaned_files(root_path: str) -> list:
    """
    Returns a list of cleaned Excel file paths from the expected folder.
    """
    folder = os.path.join(root_path, "04_Data/01_Cleansed_Files")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".xlsx")
    ]
