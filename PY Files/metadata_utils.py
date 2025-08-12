# metadata_utils.py

import os
import json
from typing import Optional
from .path_utils import get_project_paths

def get_master_metadata_path() -> str:
    """
    Returns the standardized path to master metadata index.
    
    Returns:
        str: Full path to master_metadata_index.json in metadata folder
    """
    project_paths = get_project_paths()
    return project_paths.master_metadata_path

def load_master_metadata_index(metadata_path: Optional[str] = None) -> dict:
    """
    Loads the master metadata index from JSON file.
    Uses standardized metadata folder if no path specified.

    Args:
        metadata_path (str): Optional custom path. Uses standard path if None.

    Returns:
        dict: Parsed metadata dictionary.
    """
    if metadata_path is None:
        metadata_path = get_master_metadata_path()
    
    if not os.path.exists(metadata_path):
        return {"files": []}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_master_metadata_index(metadata_index: dict, metadata_path: Optional[str] = None):
    """
    Saves the updated metadata index to disk.
    Uses standardized metadata folder if no path specified.

    Args:
        metadata_index (dict): Metadata dictionary to write.
        metadata_path (str): Optional custom path. Uses standard path if None.
    """
    if metadata_path is None:
        metadata_path = get_master_metadata_path()
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_index, f, indent=2)

def find_file_metadata(metadata_index: dict, filename: str) -> dict:
    """
    Finds metadata for a specific file.

    Args:
        metadata_index (dict): Full metadata index.
        filename (str): Filename to locate.

    Returns:
        dict or None: File-level metadata or None.
    """
    for file_meta in metadata_index.get("files", []):
        if file_meta.get("filename") == filename:
            return file_meta
    return None
