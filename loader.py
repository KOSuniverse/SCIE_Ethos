import os
import json
import os
import io
import json
import re
import pandas as pd
from typing import List, Dict, Optional

# ----------------------------
# Helpers
# ----------------------------

def _read_excel_all_sheets(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Read an Excel file (Dropbox or local) into a dict of DataFrames keyed by sheet name.
    """
    # Dropbox path heuristic: absolute path starting with "/"
    is_dbx = file_path.startswith("/")

    try:
        if is_dbx:
            from dbx_utils import read_file_bytes  # your Dropbox helper
            file_bytes = read_file_bytes(file_path)
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        else:
            return pd.read_excel(file_path, sheet_name=None)
    except ImportError:
        # If dbx_utils is not available, try local read
        return pd.read_excel(file_path, sheet_name=None)

def _read_excel_single_sheet(file_path: str, sheet_name: str, header_row_idx: Optional[int]) -> pd.DataFrame:
    """
    Read a single sheet with an optional header override.
    """
    is_dbx = file_path.startswith("/")
    try:
        if is_dbx:
            from dbx_utils import read_file_bytes
            file_bytes = read_file_bytes(file_path)
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=header_row_idx)
        else:
            return pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_idx)
    except Exception as e:
        raise

def _detect_period_from_filename(filename: str) -> str:
    m = re.search(r"Q[1-4]", filename.upper())
    return m.group(0) if m else "Unknown"

def _try_load_alias_map(file_path: str) -> dict:
    """
    Load alias map from explicit env path, sibling JSON, or Dropbox.
    Return {} on any failure.
    """
    alias_map = {}
    alias_path = os.getenv("ALIAS_PATH") or os.path.join(os.path.dirname(file_path), "global_column_aliases.json")
    try:
        from column_alias import load_alias_group
        if alias_path.startswith("/") and not os.path.exists(alias_path):
            # Try Dropbox for alias JSON
            try:
                from dbx_utils import read_file_bytes
                alias_bytes = read_file_bytes(alias_path)
                alias_map = json.loads(alias_bytes.decode("utf-8"))
            except Exception:
                alias_map = {}
        else:
            if os.path.exists(alias_path):
                alias_map = load_alias_group(alias_path)
    except Exception:
        alias_map = {}
    return alias_map

def _apply_aliases(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    """
    Apply provided alias map and AI-enhanced aliases if available.
    Silently continue on failure.
    """
    try:
        from column_alias import remap_columns, ai_enhanced_alias_builder
    except Exception:
        # If module not available, just return original df
        return df

    try:
        if alias_map:
            df = remap_columns(df, alias_map)
        # AI-enhance for unmapped columns
        enhanced_aliases, _alias_log = ai_enhanced_alias_builder(
            df, sheet_type="unknown", existing_aliases=alias_map
        )
        if enhanced_aliases:
            df = remap_columns(df, enhanced_aliases)
    except Exception:
        pass
    return df

# ----------------------------
# Header detection
# ----------------------------

def is_likely_header(row: List, df: pd.DataFrame) -> bool:
    """
    Heuristic header detector:
    - At least half of entries non-empty
    - At least half unique
    - At least half strings (non-numeric)
    """
    vals = [v for v in row if v is not None and str(v).strip()]
    total = len(row)
    if total == 0:
        return False
    unique = len(set(vals))
    str_count = sum(isinstance(v, str) and not str(v).isdigit() for v in vals)
    return (
        len(vals) >= total * 0.5 and
        unique    >= total * 0.5 and
        str_count >= total * 0.5
    )

# ----------------------------
# Public API
# ----------------------------

def load_excel_file(file_path: str, file_type: Optional[str] = None) -> List[Dict]:
    """
    Loads and tags sheets from an Excel file.

    Returns a list of records:
      {
        'df': DataFrame,
        'sheet_name': str,
        'source_file': str,
        'period': str,
        'file_type': Optional[str],
        'header_row': Optional[int],
        'alias_map_used': dict
      }
    """
    filename = os.path.basename(file_path)
    sheets = _read_excel_all_sheets(file_path)
    period = _detect_period_from_filename(filename)

    alias_map = _try_load_alias_map(file_path)

    data: List[Dict] = []
    for sheet_name, df in sheets.items():
        header_row_idx = None

        # Scan first up to 30 rows for likely header
        scan_rows = min(30, len(df))
        for i in range(scan_rows):
            row_vals = df.iloc[i].tolist()
            # Use raw values (not str-casted) for better type signal
            if is_likely_header(row_vals, df):
                header_row_idx = i
                break

        # Reload with header if detected; else best-effort fixup
        if header_row_idx is not None:
            try:
                df_valid = _read_excel_single_sheet(file_path, sheet_name, header_row_idx)
                print(f"[loader] Detected header at row {header_row_idx+1} in '{filename}' -> sheet '{sheet_name}'")
            except Exception as e:
                print(f"[loader] Reload with header failed for '{sheet_name}': {e}")
                # Fallback: promote detected row to columns
                df_valid = df.iloc[header_row_idx + 1:].copy()
                df_valid.columns = df.iloc[header_row_idx].astype(str).str.strip().tolist()
        else:
            print(f"[loader] WARNING: No header detected in '{filename}' -> sheet '{sheet_name}'. Using raw frame.")
            df_valid = df.copy()

        # Apply aliasing
        df_valid = _apply_aliases(df_valid, alias_map)

        record = {
            "df": df_valid,
            "sheet_name": sheet_name,
            "source_file": filename,
            "period": period,
            "file_type": file_type,
            "header_row": header_row_idx,
            "alias_map_used": alias_map,
        }
        data.append(record)

    return data

def load_master_metadata_index(metadata_dir: str) -> dict:
    """
    Reads the master metadata index JSON from the given directory.
    Returns {} if not found or unreadable.
    """
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

def load_files_from_folder(folder_path: str,
                           include: Optional[str] = None,
                           exclude: Optional[str] = None) -> List[Dict]:
    """
    Loads all Excel files from a folder and returns structured list.
    Filters by 'include' substring and excludes by 'exclude' substring.
    """
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".xlsx")
        and (include.lower() in f.lower() if include else True)
        and (exclude.lower() not in f.lower() if exclude else True)
    ]

    all_data: List[Dict] = []
    for file in files:
        full_path = os.path.join(folder_path, file)
        tagged = load_excel_file(full_path)
        all_data.extend(tagged)
    return all_data

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

