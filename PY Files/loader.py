import os
import json
import re
import pandas as pd


def is_likely_header(row, df):
    # Heuristic: mostly unique, mostly string, mostly non-empty
    non_empty = [v for v in row if v and str(v).strip()]
    unique = len(set(non_empty))
    str_count = sum(isinstance(v, str) and not v.isdigit() for v in non_empty)
    # Compare to total columns
    total = len(row)
    # At least half non-empty, half unique, half string
    return (
        total > 0 and
        len(non_empty) >= total * 0.5 and
        unique >= total * 0.5 and
        str_count >= total * 0.5
    )

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

    # --- Load alias map from Dropbox or local path ---
    alias_map = {}
    alias_path = os.getenv("ALIAS_PATH") or os.path.join(os.path.dirname(file_path), "global_column_aliases.json")
    try:
        from column_alias import load_alias_group, remap_columns, ai_enhanced_alias_builder
        if os.path.exists(alias_path):
            alias_map = load_alias_group(alias_path)
        else:
            # Try Dropbox if path starts with /
            if file_path.startswith('/'):
                from dbx_utils import read_file_bytes
                alias_bytes = read_file_bytes(alias_path)
                alias_map = json.loads(alias_bytes.decode('utf-8'))
    except Exception as e:
        print(f"[loader] Alias map load failed: {e}")
        alias_map = {}

    data = []
    for sheet_name, df in sheets.items():
        header_row_idx = None
        for i in range(min(30, len(df))):
            row = df.iloc[i].astype(str).str.strip().tolist()
            if is_likely_header(row, df):
                header_row_idx = i
                break
        if header_row_idx is not None:
            try:
                if file_path.startswith('/'):
                    from dbx_utils import read_file_bytes
                    import io
                    file_bytes = read_file_bytes(file_path)
                    df_valid = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=header_row_idx)
                else:
                    df_valid = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_idx)
                print(f"[loader] Detected header at row {header_row_idx+1} in sheet '{sheet_name}' of '{filename}'")
            except Exception as e:
                print(f"[loader] Failed to reload sheet '{sheet_name}' with detected header: {e}")
                df_valid = df.iloc[header_row_idx+1:].copy()
                df_valid.columns = df.iloc[header_row_idx].tolist()
        else:
            print(f"[loader] WARNING: No header detected in sheet '{sheet_name}' of '{filename}'. Ingesting all rows.")
            df_valid = df.copy()

        # --- Remap columns using alias map ---
        try:
            if alias_map:
                df_valid = remap_columns(df_valid, alias_map)
                print(f"[loader] Columns remapped for sheet '{sheet_name}' using alias map.")
            # AI harmonization for unmapped columns
            enhanced_aliases, alias_log = ai_enhanced_alias_builder(df_valid, sheet_type="unknown", existing_aliases=alias_map)
            if enhanced_aliases:
                df_valid = remap_columns(df_valid, enhanced_aliases)
                print(f"[loader] AI-enhanced alias mapping applied for sheet '{sheet_name}'. Log: {alias_log}")
        except Exception as e:
            print(f"[loader] Column alias mapping failed for sheet '{sheet_name}': {e}")

        record = {
            'df': df_valid,
            'sheet_name': sheet_name,
            'source_file': filename,
            'period': period,
            'file_type': file_type,
            'header_row': header_row_idx,
            'alias_map_used': alias_map
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

