import os, json
import re
import pandas as pd

def load_excel_file(file_path, file_type=None):
    """
    Loads and tags sheets from an Excel file.

    Args:
        file_path (str): Path to Excel file.
        file_type (str): Optional string like 'wip', 'inventory', or 'finance'.

    Returns:
        List[Dict]: Each entry contains:
            - 'df': the DataFrame
            - 'sheet_name': original sheet name
            - 'source_file': filename
            - 'period': Q1, Q2, etc.
            - 'file_type': passed type or inferred
    """
    filename = os.path.basename(file_path)
    
    # Use Dropbox API for cloud paths
    try:
        from dbx_utils import read_file_bytes
        import io
        
        # Check if this is a Dropbox path
        if file_path.startswith('/'):
            # Use Dropbox API to read the file
            file_bytes = read_file_bytes(file_path)
            sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        else:
            # Local file path - use direct pandas
            sheets = pd.read_excel(file_path, sheet_name=None)
    except ImportError:
        # Fallback to direct pandas if dbx_utils not available
        sheets = pd.read_excel(file_path, sheet_name=None)

    # Detect quarter from filename
    match = re.search(r"Q[1-4]", filename.upper())
    period = match.group(0) if match else "Unknown"

    data = []
    for sheet_name, df in sheets.items():
        record = {
            'df': df.copy(),
            'sheet_name': sheet_name,
            'source_file': filename,
            'period': period,
            'file_type': file_type
        }
        data.append(record)

    return data

# --- metadata index loader (used by orchestrator) ---

def load_master_metadata_index(metadata_dir: str) -> dict:
    """
    Reads the master metadata index JSON from the given directory.
    Returns {} if not found or unreadable.
    """
    # If your filename differs, add it here.
    candidates = [
        "master_metadata_index.json",
        "metadata_index.json",
        "master_metadata.json",
    ]
    for name in candidates:
        path = os.path.join(metadata_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def load_files_from_folder(folder_path, include=None, exclude=None):
    """
    Loads all Excel files from a folder and returns structured list.

    Args:
        folder_path (str): Directory to scan
        include (str): keyword required in filename (e.g., 'inventory')
        exclude (str): keyword to skip (e.g., 'wip')

    Returns:
        List[Dict]: list of sheet records across all matched files
    """
    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".xlsx") and
           (include.lower() in f.lower() if include else True) and
           (exclude.lower() not in f.lower() if exclude else True)
    ]

    all_data = []
    for file in files:
        full_path = os.path.join(folder_path, file)
        tagged_sheets = load_excel_file(full_path)
        all_data.extend(tagged_sheets)

    return all_data

