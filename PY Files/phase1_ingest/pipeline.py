# PY Files/phase1_ingest/pipeline.py

import os
import io
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Union, Optional, Any

# --- Import helpers from whichever module actually defines them, with safe fallbacks ---

# load_alias_group
try:
    from column_alias import load_alias_group  # preferred
except Exception:
    try:
        from alias_utils import load_alias_group  # alt in your repo
    except Exception:
        def load_alias_group(path: str) -> dict:
            """Fallback: no alias mapping available."""
            return {}

# build_reverse_alias_map
try:
    from column_alias import build_reverse_alias_map
except Exception:
    try:
        from alias_utils import build_reverse_alias_map
    except Exception:
        def build_reverse_alias_map(alias_group: dict) -> dict:
            """Fallback reverse map builder."""
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
    from column_alias import remap_columns
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
            out = df.copy()
            out.columns = [rev_lower.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
            return out

# EDA / summaries
from eda import generate_eda_summary
from summarizer import summarize_data_context

# Project helpers
from metadata_utils import save_master_metadata_index
from file_utils import list_cleaned_files
from constants import *  # if you rely on any constants

# Sheet normalization (lives in same package)
try:
    from .sheet_utils import load_sheet_aliases, normalize_sheet_type
except Exception:
    # Safe fallbacks if module moves
    def load_sheet_aliases(path: str) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    def normalize_sheet_type(sheet_name: str, df: pd.DataFrame, sheet_aliases: Optional[dict]) -> str:
        return "unclassified"

BytesLike = Union[bytes, bytearray, io.BytesIO]


# ---------------- Core helpers ----------------

def _resolve_alias_path(paths: Optional[Any]) -> Optional[str]:
    """Find alias JSON path from your paths object, or fall back to metadata folder."""
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


def _resolve_sheet_aliases(paths: Optional[Any]) -> Optional[dict]:
    """Try loading sheet_aliases.json once; ok if missing."""
    try:
        meta_dir = getattr(paths, "metadata_folder", None)
        if not meta_dir:
            return None
        sa_path = os.path.join(meta_dir, "sheet_aliases.json")
        return load_sheet_aliases(sa_path)
    except Exception:
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


def _hardened_process_excel(xls: pd.ExcelFile, filename: Optional[str], alias_path: Optional[str], paths: Optional[Any]) -> Tuple[Dict[str, pd.DataFrame], list]:
    """
    Core, production-safe processing used by both run_pipeline() and run_pipeline_on_file().
    - Never drops data
    - Adds normalized_sheet_type when possible
    - Captures per-sheet errors in metadata
    """
    cleaned_sheets: Dict[str, pd.DataFrame] = {}
    per_sheet_meta = []

    sheet_aliases = _resolve_sheet_aliases(paths)

    for sheet in xls.sheet_names:
        try:
            raw_df = pd.read_excel(xls, sheet_name=sheet)

            # 1) Clean/remap (tolerant if alias map missing)
            df = clean_and_standardize_sheet(raw_df, alias_path)
            df["source_sheet"] = sheet

            # 2) Normalize sheet type if alias file exists
            try:
                norm_type = normalize_sheet_type(sheet, df, sheet_aliases) if sheet_aliases else "unclassified"
            except Exception:
                norm_type = "unclassified"
            df["normalized_sheet_type"] = norm_type

            # 3) ALWAYS keep the data (no matter what happens next)
            cleaned_sheets[sheet] = df

            # 4) EDA & summary are best‑effort (never block output)
            eda_text = ""
            try:
                eda_text = generate_eda_summary(df)
            except Exception as e_eda:
                eda_text = f"(EDA error: {type(e_eda).__name__}: {e_eda})"

            summary_text = ""
            try:
                summary_text = summarize_data_context(eda_text, client=None)
            except Exception as e_sum:
                summary_text = f"(Summary error: {type(e_sum).__name__}: {e_sum})"

            per_sheet_meta.append({
                "filename": filename or "(unknown)",
                "sheet_name": sheet,
                "normalized_sheet_type": norm_type,
                "columns": list(df.columns),
                "record_count": int(len(df)),
                "summary_text": summary_text
            })

        except Exception as e:
            # Record failure but do not crash the whole run
            per_sheet_meta.append({
                "filename": filename or "(unknown)",
                "sheet_name": sheet,
                "error": f"{type(e).__name__}: {e}",
            })

    return cleaned_sheets, per_sheet_meta


# ---------------- Legacy disk-writing entrypoint (kept, now hardened) ----------------

def run_pipeline_on_file(xls_path, alias_path, output_prefix, output_folder):
    xls = pd.ExcelFile(xls_path)
    cleaned_sheets_dict, per_sheet_meta = _hardened_process_excel(
        xls=xls,
        filename=os.path.basename(xls_path),
        alias_path=alias_path,
        paths=None  # legacy path variant doesn't pass full paths; that's fine
    )

    # Save metadata
    metadata = {
        "run_started": datetime.utcnow().isoformat() + "Z",
        "run_completed": datetime.utcnow().isoformat() + "Z",
        "source_filename": os.path.basename(xls_path),
        "sheet_count": len(xls.sheet_names),
        "processed_sheets": list(cleaned_sheets_dict.keys()),
        "sheets": per_sheet_meta,
    }
    metadata_path = os.path.join(output_folder, f"{output_prefix}_metadata.json")
    save_master_metadata_index({"files": metadata.get("sheets", [])}, metadata_path)

    # Save cleaned file
    cleaned_file_path = os.path.join(output_folder, f"{output_prefix}_cleaned.xlsx")
    with pd.ExcelWriter(cleaned_file_path, engine="xlsxwriter") as writer:
        for sheet_name, df in cleaned_sheets_dict.items():
            writer_sheet = (sheet_name or "Sheet1")[:31]
            df.to_excel(writer, sheet_name=writer_sheet, index=False)

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

    cleaned_sheets, per_sheet_meta = _hardened_process_excel(
        xls=xls,
        filename=filename,
        alias_path=alias_path,
        paths=paths
    )

    metadata = {
        "run_started": datetime.utcnow().isoformat() + "Z",
        "run_completed": datetime.utcnow().isoformat() + "Z",
        "source_filename": filename or "(unknown)",
        "sheet_count": len(xls.sheet_names),
        "processed_sheets": list(cleaned_sheets.keys()),
        "sheets": per_sheet_meta,
    }
    return cleaned_sheets, metadata

