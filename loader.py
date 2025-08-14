# loader.py
# Robust Excel loader with deep header detection, manual promotion, and alias application

import os
import io
import re
import json
import pandas as pd
from typing import Any, Dict, List, Optional, Set

# ----------------------------
# Config
# ----------------------------
_SCAN_LIMIT = int(os.getenv("HEADER_SCAN_LIMIT", "150"))  # rows to scan for header detection

# Common tokens to help header detection (merged with alias keys)
_COMMON_HEADER_TOKENS: Set[str] = {
    "part", "part no", "part number", "item", "sku", "description", "qty", "quantity",
    "on hand", "average", "extended", "extended cost", "cost", "last used",
    "ytd", "ytd usage", "last year", "last year usage", "uom", "warehouse",
    "location", "plant", "date", "group", "family", "bin", "value", "amount",
    "price", "standard", "source_sheet", "normalized_sheet_type"
}

# ----------------------------
# Dropbox / Local Excel readers
# ----------------------------

def _read_excel_all_sheets(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    First-pass read of all sheets with header=None so we see every row exactly as in Excel.
    Works with Dropbox (via dbx_utils.read_file_bytes) or local filesystem.
    """
    is_dbx = file_path.startswith("/")
    try:
        if is_dbx:
            from dbx_utils import read_file_bytes  # user-provided helper
            b = read_file_bytes(file_path)
            return pd.read_excel(io.BytesIO(b), sheet_name=None, header=None)
        else:
            return pd.read_excel(file_path, sheet_name=None, header=None)
    except ImportError:
        # Fallback if dbx_utils is not available in this runtime
        return pd.read_excel(file_path, sheet_name=None, header=None)

# ----------------------------
# Filename period helper
# ----------------------------

def _detect_period_from_filename(filename: str) -> str:
    m = re.search(r"Q[1-4]", filename.upper())
    return m.group(0) if m else "Unknown"

# ----------------------------
# Alias loading and application
# ----------------------------

def _try_load_alias_map(file_path: str) -> dict:
    """
    Load alias map from env ALIAS_PATH, sibling JSON, or Dropbox.
    Returns {} on failure.
    """
    alias_map: dict = {}
    alias_path = os.getenv("ALIAS_PATH") or os.path.join(os.path.dirname(file_path), "global_column_aliases.json")
    try:
        from column_alias import load_alias_group  # your project helper
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
        return df  # alias helpers not available

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
# Header detection (robust, deep scan, no hardcoding)
# ----------------------------

def _is_number_like(x: Any) -> bool:
    s = str(x).strip().replace(",", "")
    if not s:
        return False
    try:
        float(s)
        return True
    except Exception:
        return False

def _row_sig(cells: List[Any]) -> Dict[str, Any]:
    vals = [c for c in cells if c is not None and str(c).strip() != ""]
    lower = [str(v).strip().lower() for v in vals]
    n = len(cells) if len(cells) > 0 else 1
    tok_len = [len(str(v).strip()) for v in vals if str(v).strip()]
    return dict(
        nonempty=len(vals)/n,
        string=sum(not _is_number_like(v) for v in vals)/max(1, len(vals)),
        numeric=sum(_is_number_like(v) for v in vals)/max(1, len(vals)),
        unique=len(set(lower))/max(1, len(lower)),
        avglen=(sum(tok_len)/len(tok_len)) if tok_len else 0.0,
        vals=vals, lower=lower
    )

def _header_keywords(alias_map: dict) -> Set[str]:
    keys = set()
    try:
        for canon, alts in alias_map.items():
            keys.add(str(canon).lower().strip().replace("_", " "))
            for a in (alts or []):
                keys.add(str(a).lower().strip())
    except Exception:
        pass
    return keys | _COMMON_HEADER_TOKENS

def _window_data_signal(df: pd.DataFrame, i: int, width: int = 3) -> float:
    """
    Measures how 'data-like' the next few rows are (more numeric and filled)
    relative to the candidate header row at index i.
    """
    rows = []
    for k in range(1, width + 1):
        if i + k < len(df):
            rows.append(_row_sig(df.iloc[i + k].tolist()))
    if not rows:
        return 0.0
    avg_num = sum(r["numeric"] for r in rows) / len(rows)
    avg_nonempty = sum(r["nonempty"] for r in rows) / len(rows)
    return 0.6 * avg_num + 0.4 * avg_nonempty

def _row_score_for_header(df: pd.DataFrame, i: int, header_words: Set[str]) -> float:
    sig = _row_sig(df.iloc[i].tolist())

    # Reject ultra-sparse or very short label rows (section titles)
    if sig["nonempty"] < 0.30 or sig["avglen"] < 3:
        return 0.0

    # Keyword hits
    hits = sum(1 for v in sig["lower"] for w in header_words if w == v or (len(w) > 3 and w in v))
    hit_ratio = hits / max(1, len(sig["lower"]))

    # Penalize repeated tokens (e.g., "total,total,total")
    if sig["lower"]:
        most = max(sig["lower"].count(x) for x in set(sig["lower"]))
        rep_ratio = most / len(sig["lower"])
    else:
        rep_ratio = 0.0

    # Data-like window below candidate (rows i+1..i+3)
    data_signal = _window_data_signal(df, i, width=3)

    # Weighted score
    base = (
        0.35 * sig["nonempty"] +
        0.25 * sig["unique"] +
        0.20 * sig["string"] +
        0.15 * hit_ratio +
        0.20 * data_signal -
        0.15 * max(0.0, rep_ratio - 0.34)
    )
    return max(0.0, base)

def detect_header_row(df: pd.DataFrame, alias_map: dict) -> Optional[int]:
    header_words = _header_keywords(alias_map)
    limit = min(_SCAN_LIMIT, len(df))
    best_idx, best_score = None, 0.0
    for i in range(limit):
        s = _row_score_for_header(df, i, header_words)
        if s > best_score:
            best_idx, best_score = i, s
    # Reasonable threshold; tune if needed
    return best_idx if (best_idx is not None and best_score >= 0.58) else None

# ----------------------------
# Column promotion & sanitization
# ----------------------------

def _build_columns_from_rows(raw_df: pd.DataFrame, hdr_idx: int) -> List[str]:
    """
    Create column names from the detected header row, borrowing from the next row
    for merged/blank cells; synthesize names when still empty.
    """
    hdr = raw_df.iloc[hdr_idx].tolist()
    nxt = raw_df.iloc[hdr_idx + 1].tolist() if hdr_idx + 1 < len(raw_df) else []
    new_cols = []
    for i, c in enumerate(hdr):
        name = str(c).strip() if c is not None else ""
        # fill blank/unnamed from next row if it looks like a label
        if not name or name.lower().startswith("unnamed"):
            if i < len(nxt):
                n_str = str(nxt[i]).strip()
                if n_str and not n_str.lower().startswith("unnamed"):
                    name = n_str
        if not name:
            name = f"col_{i+1}"
        new_cols.append(name)
    return new_cols

def _sanitize_columns(cols: List[Any]) -> List[str]:
    """
    Remove pandas artifacts and ensure uniqueness, without dropping data columns.
    """
    out, seen = [], set()
    for i, c in enumerate(cols):
        name = str(c).strip()
        if name.lower().startswith("unnamed") or name.lower() in {"nan", "none", ""}:
            name = f"col_{i+1}"
        name = re.sub(r"\s+", " ", name).strip()
        base, k = name, 2
        while name in seen:
            name = f"{base}_{k}"; k += 1
        seen.add(name)
        out.append(name)
    return out

# ----------------------------
# Public API
# ----------------------------
print("[loader] USING:", __file__)

def load_excel_file(file_path: str, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
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
    sheets = _read_excel_all_sheets(file_path)       # header=None (raw)
    period = _detect_period_from_filename(filename)
    alias_map = _try_load_alias_map(file_path)

    data: List[Dict[str, Any]] = []

    for sheet_name, df in sheets.items():
        # Detect header row (deep scan)
        header_row_idx = detect_header_row(df, alias_map)
        print(f"[loader] {filename}::{sheet_name} -> header_row_idx={header_row_idx}")

        # Manual promotion (DO NOT re-read with header=idx to avoid 'Unnamed:*' from merged cells)
        if header_row_idx is not None:
            df_valid = df.iloc[header_row_idx + 1:].copy()
            df_valid.columns = _build_columns_from_rows(df, header_row_idx)
        else:
            # Fallback: first non-empty row as header if detection fails
            first_nonempty = next(
                (i for i in range(min(_SCAN_LIMIT, len(df))) if any(str(x).strip() for x in df.iloc[i].tolist())),
                None
            )
            if first_nonempty is not None:
                df_valid = df.iloc[first_nonempty + 1:].copy()
                df_valid.columns = _build_columns_from_rows(df, first_nonempty)
            else:
                df_valid = df.copy()
                df_valid.columns = [f"col_{i+1}" for i in range(df_valid.shape[1])]

        # Drop fully-empty columns after promotion
        all_empty = df_valid.apply(lambda s: s.isna().all() or (s.astype(str).str.strip() == "").all(), axis=0)
        df_valid = df_valid.loc[:, ~all_empty]

        # Normalize/sanitize column names (removes residual Unnamed, enforces uniqueness)
        df_valid.columns = _sanitize_columns(list(df_valid.columns))

        # Apply alias mapping last (best chance after clean headers)
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

        # Optional: quick peek at columns for debugging
        print(f"[loader] {filename}::{sheet_name} -> cols[0:10]={df_valid.columns[:10].tolist()}")

    return data

# ----------------------------
# Metadata index loader (used by orchestrator)
# ----------------------------

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

# ----------------------------
# Folder loader utility
# ----------------------------

def load_files_from_folder(folder_path: str,
                           include: Optional[str] = None,
                           exclude: Optional[str] = None) -> List[Dict[str, Any]]:
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

    all_data: List[Dict[str, Any]] = []
    for file in files:
        full_path = os.path.join(folder_path, file)
        tagged = load_excel_file(full_path)
        all_data.extend(tagged)
    return all_data
