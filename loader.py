# ----------------------------
# Header detection (robust, no hardcoding)
# ----------------------------
from typing import Tuple, Set

_COMMON_HEADER_TOKENS: Set[str] = {
    "part", "part no", "part number", "item", "sku", "description", "qty", "quantity",
    "on hand", "average", "extended", "cost", "last used", "ytd", "last year",
    "uom", "warehouse", "location", "plant", "date", "group", "family", "bin"
}

def _header_keywords(alias_map: dict) -> Set[str]:
    keys = set()
    try:
        # alias_map like {"part_number": ["Part No", "Item", ...], ...}
        for canon, alts in alias_map.items():
            keys.add(str(canon).lower().strip().replace("_", " "))
            for a in (alts or []):
                keys.add(str(a).lower().strip())
    except Exception:
        pass
    return keys | _COMMON_HEADER_TOKENS

def _row_score_for_header(cells: list, next_row: list, header_words: Set[str]) -> float:
    vals = [c for c in cells if c is not None and str(c).strip() != ""]
    total = len(cells)
    if total == 0:
        return 0.0

    # basic ratios
    nonempty = len(vals) / total
    unique   = len(set(map(lambda x: str(x).strip().lower(), vals))) / max(1, len(vals))
    is_str   = sum(not str(v).replace(".", "", 1).isdigit() for v in vals) / max(1, len(vals))

    # keyword hits
    lowered = [str(v).strip().lower() for v in vals]
    hits = sum(1 for v in lowered for w in header_words if w == v or (len(w) > 3 and w in v))
    hit_ratio = hits / max(1, len(vals))

    # type switch: next row should be more numeric than header
    next_vals = [n for n in next_row if n is not None and str(n).strip() != ""]
    next_is_num = sum(str(v).replace(".", "", 1).isdigit() for v in next_vals) / max(1, len(next_vals))
    type_switch = max(0.0, next_is_num - (1 - is_str))  # reward if next row is more numeric

    # penalize single-title rows like "Dec-22"
    if len(vals) <= max(2, total * 0.2):
        return 0.0

    # weighted score
    return 0.35*nonempty + 0.25*unique + 0.2*is_str + 0.15*hit_ratio + 0.15*type_switch

def detect_header_row(df: pd.DataFrame, alias_map: dict) -> int | None:
    header_words = _header_keywords(alias_map)
    max_scan = min(50, len(df))
    best_idx, best_score = None, 0.0

    for i in range(max_scan):
        row = df.iloc[i].tolist()
        nxt = df.iloc[i+1].tolist() if i+1 < len(df) else []
        s = _row_score_for_header(row, nxt, header_words)
        if s > best_score:
            best_idx, best_score = i, s

    # threshold: must be reasonably header-like
    return best_idx if (best_idx is not None and best_score >= 0.55) else None
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
            return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)
        else:
            return pd.read_excel(file_path, sheet_name=None, header=None)
    except ImportError:
        # If dbx_utils is not available, try local read
        return pd.read_excel(file_path, sheet_name=None, header=None)

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
        header_row_idx = detect_header_row(df, alias_map)

        if header_row_idx is not None:
            try:
                df_valid = _read_excel_single_sheet(file_path, sheet_name, header_row_idx)
            except Exception:
                df_valid = df.iloc[header_row_idx + 1:].copy()
                df_valid.columns = [str(c).strip() for c in df.iloc[header_row_idx].tolist()]
        else:
            # fallback to first non-empty row as header if present
            first_nonempty = next((i for i in range(min(50, len(df)))
                                   if any(str(x).strip() for x in df.iloc[i].tolist())), None)
            if first_nonempty is not None:
                try:
                    df_valid = _read_excel_single_sheet(file_path, sheet_name, first_nonempty)
                except Exception:
                    df_valid = df.iloc[first_nonempty + 1:].copy()
                    df_valid.columns = [str(c).strip() for c in df.iloc[first_nonempty].tolist()]
            else:
                df_valid = df.copy()

        # clean column names
        df_valid.columns = [
            None if str(c).lower().startswith("unnamed") else str(c).strip()
            for c in df_valid.columns]
        df_valid = df_valid.loc[:, [c for c in df_valid.columns if c]]

        # apply aliases
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

