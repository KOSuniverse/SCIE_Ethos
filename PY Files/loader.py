import os
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

