# PY Files/phase1_ingest/pipeline.py

# pipeline.py
import os
import io
import pandas as pd
from typing import Dict, Any, List, Tuple

# Robust header handling from your loader
from loader import detect_header_row, _build_columns_from_rows, _sanitize_columns

# Classifier / cleaning / EDA / summaries (use your modules; graceful fallbacks)
try:
    from sheet_utils import classify_sheet
except Exception:
    def classify_sheet(sheet_name, df, sheet_aliases=None):
        return {"final_type": "unclassified"}

try:
    from smart_autofix_system import run_smart_autofix
except Exception:
    def run_smart_autofix(df, sheet_name="", aggressive_mode=False):
        return df, [], {"note": "smart_autofix unavailable; returning df unchanged"}

try:
    from enhanced_eda_system import run_enhanced_eda
except Exception:
    def run_enhanced_eda(df, sheet_name="", filename=""):
        return {"metadata": {"rows": len(df), "cols": len(df.columns)}, "chart_paths": [], "business_insights": {}}

try:
    from gpt_summary_generator import generate_comprehensive_summary
except Exception:
    def generate_comprehensive_summary(df, sheet_name, filename, metadata=None, eda_results=None, cleaning_log=None):
        return {"summary": f"{filename}::{sheet_name} rows={len(df)} cols={len(df.columns)}"}

# Per-sheet artifact writers + rollups (XLSX only in save_cleansed_table)
from metadata_utils import (
    save_per_sheet_metadata, save_per_sheet_summary, save_per_sheet_eda,
    save_cleansed_table, rollup_all_master_indexes
)

# ----- helpers -----

def _first_nonempty_row(df: pd.DataFrame, limit: int = 150) -> int | None:
    limit = min(limit, len(df))
    for i in range(limit):
        row = df.iloc[i].tolist()
        if any(str(x).strip() for x in row):
            return i
    return None

def _promote_headers(df0: pd.DataFrame, sheet_name: str, filename: str) -> pd.DataFrame:
    """Detect header row, promote manually, sanitize columns. df0 is read with header=None."""
    alias_map = {}
    hdr = detect_header_row(df0, alias_map)
    if hdr is None:
        hdr = _first_nonempty_row(df0, limit=300)

    if hdr is not None:
        df = df0.iloc[hdr + 1:].copy()
        df.columns = _build_columns_from_rows(df0, hdr)
    else:
        df = df0.copy()
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]

    # drop fully-empty columns
    empty_mask = df.apply(lambda s: s.isna().all() or (s.astype(str).str.strip() == "").all(), axis=0)
    df = df.loc[:, ~empty_mask]
    # sanitize names
    df.columns = _sanitize_columns(list(df.columns))
    return df

# ----- public API -----

def run_pipeline(source: str | bytes, filename: str, paths: Dict[str, Any] | None = None) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    """
    Reads ALL sheets in an Excel file (path or bytes), promotes headers, classifies, cleans,
    writes per-sheet metadata/summaries/eda, and saves cleansed tables split by type (XLSX only).
    Returns (cleaned_sheets, per_sheet_meta_list).
    """
    # Handle path-like vs bytes vs file-like for ExcelFile
    if isinstance(source, (str, os.PathLike)):
        xls = pd.ExcelFile(source)
    elif isinstance(source, bytes):
        xls = pd.ExcelFile(io.BytesIO(source))  # avoid FutureWarning
    else:
        # Assume file-like object
        xls = pd.ExcelFile(source)

    cleaned_sheets: Dict[str, pd.DataFrame] = {}
    per_sheet_meta: List[Dict[str, Any]] = []

    for sheet in xls.sheet_names:
        # 1) read raw (always header=None)
        df0 = xls.parse(sheet, header=None)
        # 2) promote & sanitize
        df = _promote_headers(df0, sheet, filename)

        # 3) classify -> normalized type (inventory/wip/unclassified)
        try:
            cls = classify_sheet(sheet, df, sheet_aliases={})
            stype = (cls.get("final_type") or "unclassified").lower()
        except Exception as e:
            # defensive fallback for any pandas.Index truthiness errors or classifier crashes
            msg = str(e).lower()
            cols = [str(c).strip().lower() for c in df.columns]
            has_part = any(k in c for c in cols for k in ("part", "sku", "item"))
            has_job  = any(k in c for c in cols for k in ("job", "work order", "wo"))
            if "truth value of a index is ambiguous" in msg:
                if has_part and not has_job: stype = "inventory"
                elif has_job and not has_part: stype = "wip"
                elif has_part and has_job: stype = "inventory"
                else: stype = "unclassified"
            else:
                # last-resort fallback
                stype = "inventory" if has_part else ("wip" if has_job else "unclassified")

        # 4) clean
        df_clean, ops_log, clean_report = run_smart_autofix(df, sheet_name=sheet, aggressive_mode=False)

        # 5) RAW metadata (after header promotion)
        save_per_sheet_metadata(filename, sheet, {
            "stage": "raw",
            "normalized_sheet_type": stype,
            "columns": list(map(str, df.columns)),
            "row_count": int(len(df))
        }, df=df)

        # 6) CLEANSed table (split by type) -> XLSX only
        out_path = save_cleansed_table(df_clean, filename, sheet, normalized_type=stype)

        # 7) CLEANSed metadata
        save_per_sheet_metadata(filename, sheet, {
            "stage": "cleansed",
            "normalized_sheet_type": stype,
            "columns": list(map(str, df_clean.columns)),
            "row_count": int(len(df_clean)),
            "output_path": out_path
        }, df=df_clean)

        # 8) EDA + persist
        eda_doc = run_enhanced_eda(df_clean, sheet_name=sheet, filename=filename)
        save_per_sheet_eda(filename, sheet, eda_doc.get("metadata", {}) | {
            "chart_paths": eda_doc.get("chart_paths", []),
            "business_insights": eda_doc.get("business_insights", {})
        })

        # 9) Summaries + persist
        sums = generate_comprehensive_summary(
            df_clean, sheet, filename,
            metadata=eda_doc.get("metadata"), eda_results=eda_doc, cleaning_log=ops_log
        )
        if isinstance(sums, dict):
            for k, v in sums.items():
                save_per_sheet_summary(filename, sheet, v, stage="cleansed", kind=k)
        else:
            save_per_sheet_summary(filename, sheet, str(sums), stage="cleansed", kind="summary")

        # 10) accumulate for caller
        cleaned_sheets[sheet] = df_clean
        per_sheet_meta.append({
            "filename": filename,
            "sheet_name": sheet,
            "normalized_sheet_type": stype,
            "rows": len(df_clean),
            "columns": list(map(str, df_clean.columns)),
            "output_path": out_path
        })

    # 11) update master indexes once per file
    rollup_all_master_indexes()
    return cleaned_sheets, per_sheet_meta

def run_pipeline_on_file(xls_path: str, alias_path: str | None = None, output_prefix: str | None = None, output_folder: str | None = None):
    """Thin wrapper for code that previously called this single-file entrypoint."""
    cleaned_sheets, per_sheet_meta = run_pipeline(source=xls_path, filename=os.path.basename(xls_path), paths=None)
    total_rows = sum(m["rows"] for m in per_sheet_meta)
    any_cols = len(next(iter(cleaned_sheets.values())).columns) if cleaned_sheets else 0
    return {
        "cleansed_ok": True,
        "validation_issues": [],
        "sheet_summaries": {"sheets": len(per_sheet_meta), "rows": total_rows, "columns": any_cols},
        "cleansed_paths": [m["output_path"] for m in per_sheet_meta if "output_path" in m],
    }
