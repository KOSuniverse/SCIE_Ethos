# pipeline.py
import os
import io
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable

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
        # trivial fallback text if GPT not available
        return {"summary": f"{filename}::{sheet_name} rows={len(df)} cols={len(df.columns)}"}

# Per-sheet artifact writers + rollups (XLSX only in save_cleansed_table)
from metadata_utils import (
    save_per_sheet_metadata, save_per_sheet_summary, save_per_sheet_eda,
    save_cleansed_table, rollup_all_master_indexes
)

AMBIG_MSG = "truth value of a index is ambiguous"

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

def _fallback_type_from_columns(df: pd.DataFrame) -> str:
    cols = [str(c).strip().lower() for c in df.columns]
    has_part = any(k in c for c in cols for k in ("part", "sku", "item"))
    has_job  = any(k in c for c in cols for k in ("job", "work order", "wo"))
    if has_part and not has_job: return "inventory"
    if has_job and not has_part: return "wip"
    if has_part and has_job:     return "inventory"
    return "unclassified"

# ----- public API -----

def run_pipeline(
    source: str | bytes,
    filename: str,
    paths: Dict[str, Any] | None = None,
    reporter: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    """
    Reads ALL sheets in an Excel file (path or bytes), promotes headers, classifies, cleans,
    writes per-sheet metadata/summaries/eda, and saves cleansed tables split by type (XLSX only).
    Emits reporter events if provided.
    Returns (cleaned_sheets, per_sheet_meta_list).
    """
    def _report(ev: str, payload: Dict[str, Any]):
        if reporter:
            try:
                reporter(ev, payload)
            except Exception:
                pass

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

    _report("start_file", {"filename": filename, "sheets": len(xls.sheet_names)})

    for idx, sheet in enumerate(xls.sheet_names, start=1):
        errors: List[str] = []
        _report("sheet_start", {"sheet": sheet, "i": idx, "n": len(xls.sheet_names)})

        # 1) read raw (always header=None)
        try:
            df0 = xls.parse(sheet, header=None)
            _report("read_ok", {"sheet": sheet})
        except Exception as e:
            errors.append(f"read_error: {e}")
            save_per_sheet_metadata(filename, sheet, {
                "stage": "raw",
                "normalized_sheet_type": stype,
                "columns": [str(c) for c in df_clean.columns],      # <-- FIX
                "row_count": int(len(df_clean)),
                "output_path": out_path,
                "errors": errors[:]
            }, df=df)
            _report("sheet_done", {"sheet": sheet, "errors": errors})
            continue

        # 2) promote & sanitize
        try:
            df = _promote_headers(df0, sheet, filename)
            _report("promoted", {"sheet": sheet, "cols": len(df.columns)})
        except Exception as e:
            errors.append(f"promote_error: {e}")
            df = df0.copy()
            df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
            _report("promoted", {"sheet": sheet, "cols": len(df.columns), "fallback": True})

        # 3) classify -> normalized type
        try:
            cls = classify_sheet(sheet, df, sheet_aliases={})
            stype = (cls.get("final_type") or "unclassified").lower()
            _report("classified", {"sheet": sheet, "type": stype})
        except Exception as e:
            errors.append(f"classify_error: {e}")
            stype = _fallback_type_from_columns(df)
            _report("classified", {"sheet": sheet, "type": stype, "fallback": True})

        # 4) clean (hardened)
        try:
            df_clean, ops_log, clean_report = run_smart_autofix(df, sheet_name=sheet, aggressive_mode=False)
            _report("cleaned", {"sheet": sheet, "rows": len(df_clean), "cols": len(df_clean.columns)})
        except Exception as e:
            errors.append(f"clean_error: {e}")
            df_clean, ops_log, clean_report = df, [], {"note": "cleaning failed; using uncleaned df"}
            _report("cleaned", {"sheet": sheet, "rows": len(df_clean), "cols": len(df_clean.columns), "fallback": True})

        # 5) RAW metadata (after header promotion)
        try:
            save_per_sheet_metadata(filename, sheet, {
                "stage": "raw",
                "normalized_sheet_type": stype,
                "columns": [str(c) for c in df.columns],  # explicit list
                "row_count": int(len(df)),
                "errors": errors[:]  # snapshot so far
            }, df=df)
        except Exception as e:
            errors.append(f"raw_meta_error: {e}")

        # 6) CLEANSed table (split by type) -> XLSX only
        try:
            out_path = save_cleansed_table(df_clean, filename, sheet, normalized_type=stype)
            _report("saved", {"sheet": sheet, "path": out_path or ""})
        except Exception as e:
            errors.append(f"save_cleansed_error: {e}")
            out_path = None
            _report("saved", {"sheet": sheet, "path": "", "error": str(e)})

        # 7) CLEANSed metadata
        try:
            save_per_sheet_metadata(filename, sheet, {
                "stage": "cleansed",
                "normalized_sheet_type": stype,
                "columns": [str(c) for c in df_clean.columns],
                "row_count": int(len(df_clean)),
                "output_path": out_path,
                "errors": errors[:]
            }, df=df_clean)
        except Exception as e:
            errors.append(f"cleansed_meta_error: {e}")

        # 8) EDA + persist (hardened)
        eda_doc = None
        try:
            eda_doc = run_enhanced_eda(df_clean, sheet_name=sheet, filename=filename)
            save_per_sheet_eda(filename, sheet, eda_doc.get("metadata", {}) | {
                "chart_paths": eda_doc.get("chart_paths", []),
                "business_insights": eda_doc.get("business_insights", {}),
                "errors": errors[:]
            })
            _report("eda_done", {"sheet": sheet})
        except Exception as e:
            errors.append(f"eda_error: {e}")
            try:
                save_per_sheet_eda(filename, sheet, {
                    "rows": len(df_clean),
                    "cols": len(df_clean.columns),
                    "fallback": True,
                    "errors": errors[:]
                })
                _report("eda_done", {"sheet": sheet, "fallback": True})
            except Exception as e2:
                errors.append(f"eda_meta_error: {e2}")

        # 9) Summaries + persist (hardened, GPT/Assistant path in gpt_summary_generator)
        try:
            sums = generate_comprehensive_summary(
                df_clean, sheet, filename,
                metadata=eda_doc.get("metadata") if isinstance(eda_doc, dict) else None,
                eda_results=eda_doc if isinstance(eda_doc, dict) else None,
                cleaning_log=ops_log
            )
            if isinstance(sums, dict):
                for k, v in sums.items():
                    save_per_sheet_summary(filename, sheet, v, stage="cleansed", kind=k)
            else:
                save_per_sheet_summary(filename, sheet, str(sums), stage="cleansed", kind="summary")
            _report("summary_done", {"sheet": sheet})
        except Exception as e:
            errors.append(f"summary_error: {e}")
            try:
                save_per_sheet_summary(
                    filename, sheet,
                    f"# Summary (fallback)\n\nFile: {filename}\nSheet: {sheet}\nRows={len(df_clean)} Cols={len(df_clean.columns)}\n\nErrors: {errors}",
                    stage="cleansed", kind="summary_fallback"
                )
                _report("summary_done", {"sheet": sheet, "fallback": True})
            except Exception as e2:
                errors.append(f"summary_meta_error: {e2}")

        # 10) accumulate for caller
        cleaned_sheets[sheet] = df_clean
        per_sheet_meta.append({
            "filename": filename,
            "sheet_name": sheet,
            "normalized_sheet_type": stype,
            "rows": len(df_clean),
            "columns": list(map(str, df_clean.columns)),
            "output_path": out_path,
            "errors": errors
        })

        _report("sheet_done", {"sheet": sheet, "errors": errors})

    # 11) update master indexes once per file
    rollup_all_master_indexes()
    _report("file_done", {"filename": filename})
    return cleaned_sheets, per_sheet_meta


def run_pipeline_on_file(
    xls_path: str,
    alias_path: str | None = None,
    output_prefix: str | None = None,
    output_folder: str | None = None,
    reporter: Optional[Callable[[str, Dict[str, Any]], None]] = None,
):
    """Thin wrapper kept for legacy callers; now supports reporter too."""
    cleaned_sheets, per_sheet_meta = run_pipeline(
        source=xls_path,
        filename=os.path.basename(xls_path),
        paths=None,
        reporter=reporter,
    )
    total_rows = sum(m["rows"] for m in per_sheet_meta)
    any_cols = len(next(iter(cleaned_sheets.values())).columns) if cleaned_sheets else 0
    return {
        "cleansed_ok": True,
        "validation_issues": [],
        "sheet_summaries": {"sheets": len(per_sheet_meta), "rows": total_rows, "columns": any_cols},
        "cleansed_paths": [m["output_path"] for m in per_sheet_meta if "output_path" in m],
    }
