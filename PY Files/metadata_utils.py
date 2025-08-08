# metadata_utils.py

import os
import json

def load_master_metadata_index(metadata_path: str) -> dict:
    """
    Loads the master metadata index from JSON file.

    Args:
        metadata_path (str): Path to metadata JSON.

    Returns:
        dict: Parsed metadata dictionary.
    """
    if not os.path.exists(metadata_path):
        return {"files": []}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_master_metadata_index(metadata_index: dict, metadata_path: str):
    """
    Saves the updated metadata index to disk.

    Args:
        metadata_index (dict): Metadata dictionary to write.
        metadata_path (str): Destination path.
    """
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
