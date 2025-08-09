# PY Files/phase1_ingest/pipeline.py

import os
import io
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Union, Optional, Any

# --- Import helpers from whichever module actually defines them, with safe fallbacks ---

# load_alias_group
try:
    from column_alias import load_alias_group  # preferred location in your tree
except Exception:
    try:
        from alias_utils import load_alias_group  # alt location found in your repo
    except Exception:
        def load_alias_group(path: str) -> dict:
            """Fallback: no alias mapping available."""
            return {}

# build_reverse_alias_map
try:
    from column_alias import build_reverse_alias_map  # may not exist in your file
except Exception:
    try:
        from alias_utils import build_reverse_alias_map
    except Exception:
        def build_reverse_alias_map(alias_group: dict) -> dict:
            """
            Fallback: build reverse map like {"qty on hand": "on_hand_qty", ...}
            from alias_group shaped like {"on_hand_qty": ["qty on hand", "qoh", ...], ...}
            """
            rev = {}
            for canonical, synonyms in (alias_group or {}).items():
                key = str(canonical).strip()
                rev[key] = key
                if isinstance(synonyms, (list, tuple, set)):
                    for s in synonyms:
                        rev[str(s).strip()] = key
                elif synonyms is not None:
                    rev[str(synonyms).strip()] = key
            return rev

# remap_columns
try:
    from column_alias import remap_columns  # may not exist in your file
except Exception:
    try:
        from alias_utils import remap_columns
    except Exception:
        def remap_columns(df: pd.DataFrame, reverse_map: dict) -> pd.DataFrame:
            """Fallback: case-insensitive rename using reverse_map keys."""
            if not reverse_map:
                return df.copy()
            rev_lower = {str(k).strip().lower(): str(v).strip()
                         for k, v in reverse_map.items()}
            new_cols = []
            for c in df.columns:
                key = str(c).strip()
                mapped = rev_lower.get(key.lower(), key)
                new_cols.append(mapped)
            out = df.copy()
            out.columns = new_cols
            return out

# These should exist in your repo per screenshots; leave as-is.
from metadata_utils import save_master_metadata_index
from summarizer import summarize_data_context
from eda import generate_eda_summary
from file_utils import list_cleaned_files
from constants import *  # if you rely on any constants

BytesLike = Union[bytes, bytearray, io.BytesIO]

# ---------------- Core helpers ----------------

def _resolve_alias_path(paths: Optional[Any]) -> Optional[str]:
    """Find the alias JSON path from your paths object, or fall back to metadata folder."""
    if paths is None:
        return None
    for attr in ("alias_json", "ALIAS_JSON", "alias_path"):
        if hasattr(paths, attr):
            val = getattr(paths, attr)
            if isinstance(val, str) and val:
                return val
    if hasattr(paths, "metadata_folder"):
        return os.path.join(paths.metadata_folder, "global_column_aliases.json")
    return None


def clean_and_standardize_sheet(sheet_df: pd.DataFrame, alias_path: Optional[str]) -> pd.DataFrame:
    """
    Remap columns using alias map if available; otherwise return a copy unchanged.
    Fully tolerant of missing/invalid alias map.
    """
    df = sheet_df.copy()
    if not alias_path or not os.path.basename(alias_path):
        return df
    try:
        alias_group = load_alias_group(alias_path)  # {} if fallback
        if alias_group:
            reverse_map = build_reverse_alias_map(alias_group)
            df = remap_columns(df, reverse_map)
    except Exception:
        # If alias load/remap fails, proceed with original columns
        pass
    return df


# ---------------- Legacy disk-writing entrypoint (kept intact) ----------------

def run_pipeline_on_file(xls_path, alias_path, output_prefix, output_folder):
    xls = pd.ExcelFile(xls_path)
    cleaned_sheets = []
    all_metadata = []

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            df = clean_and_standardize_sheet(df, alias_path)
            df["source_sheet"] = sheet

            eda_text = generate_eda_summary(df)
            summary_text = summarize_data_context(eda_text, client=None)
            metadata_entry = {
                "filename": os.path.basename(xls_path),
                "sheet_name": sheet,
                "columns": list(df.columns),
                "record_count": int(len(df)),
                "sheet_type": "unclassified",
                "summary_text": summary_text,
            }
            all_metadata.append(metadata_entry)
            cleaned_sheets.append(df)
        except Exception as e:
            print(f"⚠️ Failed to process sheet '{sheet}': {e}")

    # Save metadata
    metadata_path = os.path.join(output_folder, f"{output_prefix}_metadata.json")
    save_master_metadata_index({"files": all_metadata}, metadata_path)

    # Save cleaned file
    cleaned_file_path = os.path.join(output_folder, f"{output_prefix}_cleaned.xlsx")
    with pd.ExcelWriter(cleaned_file_path, engine="xlsxwriter") as writer:
        for df in cleaned_sheets:
            name = df["source_sheet"].iloc[0]
            df.to_excel(writer, sheet_name=name[:31], index=False)

    return cleaned_file_path, metadata_path


# ---------------- In‑memory wrapper for Streamlit/Dropbox ----------------

def run_pipeline(
    source: Union[str, BytesLike],
    filename: Optional[str] = None,
    paths: Optional[Any] = None
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """
    Returns in-memory cleaned sheets + metadata (no local writes).
    Keeps your legacy run_pipeline_on_file intact.
    """
    # Open Excel from path or bytes
    if isinstance(source, (bytes, bytearray)):
        xls = pd.ExcelFile(io.BytesIO(source))
    elif isinstance(source, io.BytesIO):
        xls = pd.ExcelFile(source)
    elif isinstance(source, str):
        xls = pd.ExcelFile(source)
        if filename is None:
            filename = os.path.basename(source)
    else:
        raise TypeError("Unsupported source type for run_pipeline")

    alias_path = _resolve_alias_path(paths)

    cleaned_sheets: Dict[str, pd.DataFrame] = {}
    per_sheet_meta = []

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            df = clean_and_standardize_sheet(df, alias_path)
            df["source_sheet"] = sheet

            eda_text = generate_eda_summary(df)
            summary_text = summarize_data_context(eda_text, client=None)

            per_sheet_meta.append({
                "filename": filename or "(unknown)",
                "sheet_name": sheet,
                "columns": list(df.columns),
                "record_count": int(len(df)),
                "sheet_type": "unclassified",
                "summary_text": summary_text,
            })
            cleaned_sheets[sheet] = df
        except Exception as e:
            per_sheet_meta.append({
                "filename": filename or "(unknown)",
                "sheet_name": sheet,
                "error": f"{type(e).__name__}: {e}",
            })

    metadata = {
        "run_started": datetime.utcnow().isoformat() + "Z",
        "run_completed": datetime.utcnow().isoformat() + "Z",
        "source_filename": filename or "(unknown)",
        "sheet_count": len(xls.sheet_names),
        "processed_sheets": list(cleaned_sheets.keys()),
        "sheets": per_sheet_meta,
    }
    return cleaned_sheets, metadata
