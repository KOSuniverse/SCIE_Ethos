# --- Robust header detection (handles deep headers, no date assumption) ---

# --- Imports for robust header detection ---
import os
import math
import datetime as _dt
import pandas as pd
import io
from typing import List, Set, Optional, Dict, Any

_SCAN_LIMIT = int(os.getenv("HEADER_SCAN_LIMIT", "150"))  # configurable

_COMMON_HEADER_TOKENS: Set[str] = {
    "part", "part no", "part number", "item", "sku", "description", "qty", "quantity",
    "on hand", "average", "extended", "extended cost", "cost", "last used", "ytd",
    "ytd usage", "last year", "last year usage", "uom", "warehouse", "location",
    "plant", "date", "group", "family", "bin", "value", "amount", "price", "standard"
}

def _header_keywords(alias_map: Dict[str, List[str]]) -> Set[str]:
    keys = set()
    try:
        for canon, alts in alias_map.items():
            keys.add(str(canon).lower().strip().replace("_", " "))
            for a in (alts or []):
                keys.add(str(a).lower().strip())
    except Exception:
        pass
    return keys | _COMMON_HEADER_TOKENS

def _is_number_like(x: Any) -> bool:
    s = str(x).strip()
    if not s:
        return False
    s2 = s.replace(",", "")
    try:
        float(s2)
        return True
    except Exception:
        return False

def _token_len_avg(vals: List[str]) -> float:
    toks = [len(str(v).strip()) for v in vals if str(v).strip()]
    return (sum(toks) / len(toks)) if toks else 0.0

def _row_signature(cells: List[Any]) -> Dict[str, Any]:
    vals = [c for c in cells if c is not None and str(c).strip() != ""]
    lowered = [str(v).strip().lower() for v in vals]
    n = len(cells) if len(cells) > 0 else 1
    return {
        "nonempty_ratio": len(vals) / n,
        "string_ratio": sum(not _is_number_like(v) for v in vals) / max(1, len(vals)),
        "numeric_ratio": sum(_is_number_like(v) for v in vals) / max(1, len(vals)),
        "unique_ratio": len(set(lowered)) / max(1, len(lowered)),
        "avg_token_len": _token_len_avg(vals),
        "vals": vals, "lowered": lowered
    }

def _window_data_signal(df: Any, start_row: int, width: int = 3) -> float:
    rows = []
    for k in range(1, width+1):
        if start_row + k < len(df):
            rows.append(_row_signature(df.iloc[start_row + k].tolist()))
    if not rows:
        return 0.0
    avg_num = sum(r["numeric_ratio"] for r in rows) / len(rows)
    avg_nonempty = sum(r["nonempty_ratio"] for r in rows) / len(rows)
    return 0.6 * avg_num + 0.4 * avg_nonempty

def _row_score_for_header(df: Any, i: int, header_words: Set[str]) -> float:
    sig = _row_signature(df.iloc[i].tolist())
    if sig["nonempty_ratio"] < 0.3 or sig["avg_token_len"] < 3:
        return 0.0
    base = (
        0.35 * sig["nonempty_ratio"] +
        0.25 * sig["unique_ratio"] +
        0.20 * sig["string_ratio"] +
        0.05 * (1.0 - sig["numeric_ratio"])
    )
    hits = sum(1 for v in sig["lowered"] for w in header_words if w == v or (len(w) > 3 and w in v))
    hit_ratio = hits / max(1, len(sig["lowered"]))
    base += 0.15 * hit_ratio
    if len(sig["lowered"]) >= 3:
        most_common_freq = max(sig["lowered"].count(x) for x in set(sig["lowered"]))
        rep_ratio = most_common_freq / len(sig["lowered"])
        base -= 0.15 * max(0.0, rep_ratio - 0.34)
    data_signal = _window_data_signal(df, i, width=3)
    base += 0.20 * data_signal
    return max(0.0, base)

def detect_header_row(df: Any, alias_map: Dict[str, List[str]]) -> Optional[int]:
    header_words = _header_keywords(alias_map)
    max_scan = min(_SCAN_LIMIT, len(df))
    best_idx, best_score = None, 0.0
    for i in range(max_scan):
        s = _row_score_for_header(df, i, header_words)
        if s > best_score:
            best_idx, best_score = i, s
    return best_idx if (best_idx is not None and best_score >= 0.58) else None


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
    def _build_columns_from_rows(raw_df, hdr_idx: int):
        hdr = raw_df.iloc[hdr_idx].tolist()
        nxt = raw_df.iloc[hdr_idx + 1].tolist() if hdr_idx + 1 < len(raw_df) else []
        new_cols = []
        for i, c in enumerate(hdr):
            c_str = str(c).strip() if c is not None else ""
            # fill blanks/Unnamed from the next row if it looks like a label
            if not c_str or c_str.lower().startswith("unnamed"):
                if i < len(nxt):
                    n_str = str(nxt[i]).strip()
                    if n_str and not n_str.lower().startswith("unnamed"):
                        c_str = n_str
            # still empty? synthesize a name
            if not c_str:
                c_str = f"col_{i+1}"
            new_cols.append(c_str)
        return new_cols

    for sheet_name, df in sheets.items():
        header_row_idx = detect_header_row(df, alias_map)
        print(f"[loader] {filename} :: {sheet_name} -> header_row_idx={header_row_idx}")

        if header_row_idx is not None:
            # Manual promote (don’t re-read the sheet)
            df_valid = df.iloc[header_row_idx + 1:].copy()
            df_valid.columns = _build_columns_from_rows(df, header_row_idx)
        else:
            # fallback: first non-empty row
            first_nonempty = next((i for i in range(min(_SCAN_LIMIT, len(df)))
                                   if any(str(x).strip() for x in df.iloc[i].tolist())), None)
            if first_nonempty is not None:
                df_valid = df.iloc[first_nonempty + 1:].copy()
                df_valid.columns = _build_columns_from_rows(df, first_nonempty)
            else:
                df_valid = df.copy()

        # Drop columns that are entirely empty after promotion
        all_empty = df_valid.apply(lambda s: s.isna().all() or (s.astype(str).str.strip() == "").all(), axis=0)
        df_valid = df_valid.loc[:, ~all_empty]

        # Normalize column names; remove residual Unnamed:* names if the column is empty
        clean_cols = []
        for c in df_valid.columns:
            name = str(c).strip()
            if name.lower().startswith("unnamed"):
                name = ""  # mark for removal if empty-only; otherwise keep synthesized col_*
            clean_cols.append(name)
        df_valid.columns = clean_cols
        df_valid = df_valid.loc[:, [c for c in df_valid.columns if c]]

        # Apply aliases last
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


