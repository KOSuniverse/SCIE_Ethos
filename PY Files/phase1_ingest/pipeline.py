# PY Files/phase1_ingest/pipeline.py

import os
import io
import re
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

# EDA / summaries (best-effort)
try:
    from eda import generate_eda_summary  # your module
except Exception:
    def generate_eda_summary(df: pd.DataFrame) -> str:
        try:
            return f"(EDA) rows={len(df)}, cols={len(df.columns)}"
        except Exception as e:
            return f"(EDA error: {e})"

# Project helpers
try:
    from metadata_utils import save_master_metadata_index
except Exception:
    def save_master_metadata_index(data: dict, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

try:
    from file_utils import list_cleaned_files
except Exception:
    def list_cleaned_files(*args, **kwargs):  # not used here
        return []

try:
    from constants import *  # if you rely on any constants
except Exception:
    pass

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

# LLM client (for GPT summaries)
try:
    from llm_client import get_openai_client, chat_completion
except Exception:
    get_openai_client = None
    chat_completion = None


BytesLike = Union[bytes, bytearray, io.BytesIO]

# ---------------- Heuristics ----------------

def _detect_sheet_type_by_columns(columns, filename: str) -> str:
    cols = {str(c).strip().lower() for c in columns}
    fname = (filename or "").lower()

    # Inventory signals
    inv_keys_any = {"part_number", "part_no", "item", "item_number"}
    inv_vals_any = {"value", "inventory_value", "extended_cost", "total_cost"}
    inv_qty_any  = {"remaining_quantity", "qty_on_hand", "qoh", "quantity", "on_hand_qty"}

    if (cols & inv_keys_any) and (cols & inv_qty_any) and (cols & inv_vals_any):
        return "inventory"

    # WIP signals
    wip_job_any = {"job", "job_no", "job_number", "work_order", "wo"}
    wip_bucket_any = {
        "0–30_days","0-30_days","31–60_days","31-60_days",
        "61–90_days","61-90_days",">90_days","over_90","over 90","90+"
    }
    if (cols & wip_job_any) or (cols & wip_bucket_any) or ("wip" in fname) or ("aged wip" in fname):
        return "wip"

    # Finished goods / inventory variants
    if "inventory" in fname or "stock" in fname:
        return "inventory"

    return "unclassified"


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


def _clean_and_standardize_sheet(sheet_df: pd.DataFrame, alias_path: Optional[str]) -> pd.DataFrame:
    """
    Remap columns using alias map if available; otherwise return a copy unchanged.
    Fully tolerant of missing/invalid alias map.
    """
    df = sheet_df.copy()
    # normalize headers to snake-ish (preserve existing names if remap will handle)
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]

    if not alias_path or not os.path.basename(alias_path):
        return df
    try:
        alias_group = load_alias_group(alias_path)  # {} if fallback
        if alias_group:
            reverse_map = build_reverse_alias_map(alias_group)
            df = remap_columns(df, reverse_map)
    except Exception:
        pass
    return df


def _summarize_inventory_fallback(df: pd.DataFrame) -> str:
    try:
        n = len(df)
        cols = ", ".join(map(str, df.columns[:6])) + ("..." if df.shape[1] > 6 else "")
        value_col = None
        for cand in ["value", "extended_cost", "inventory_value", "total_cost"]:
            if cand in df.columns:
                value_col = cand
                break
        value_txt = ""
        if value_col is not None and pd.api.types.is_numeric_dtype(df[value_col]):
            total_val = float(pd.to_numeric(df[value_col], errors="coerce").fillna(0).sum())
            value_txt = f"  • Total {value_col}: ${total_val:,.0f}\n"
        return (
            f"Inventory sheet detected.\n"
            f"  • Rows: {n}\n"
            f"  • Example columns: {cols}\n"
            f"{value_txt}"
            f"Common checks: low usage-to-stock, high value concentration, negative qty."
        ).strip()
    except Exception as e:
        return f"(Fallback summary unavailable: {e})"


def _gpt_summary_for_sheet(client, sheet_type: str, filename: str, df: pd.DataFrame, eda_text: str = "") -> str:
    # keep payload small & safe
    try:
        sample = df.head(20).to_dict(orient="records")
    except Exception:
        sample = []

    # Coerce EDA text safely
    try:
        eda_snippet = str(eda_text)
    except Exception:
        eda_snippet = ""
    if len(eda_snippet) > 600:
        eda_snippet = eda_snippet[:600]

    prompt = (
        "You are a precise data analyst. Briefly summarize this sheet for a run log.\n"
        f"- Sheet type guess: {sheet_type}\n"
        f"- Filename: {filename}\n"
        "- Focus on: row count, notable columns, and any obvious metrics (e.g., total value if present).\n"
        "- Keep it under 120 words.\n"
        f"\nEDA notes (may be empty): {eda_snippet}\n"
        f"\nSample rows (JSON): {json.dumps(sample, default=str)[:3500]}"
    )
    messages = [
        {"role": "system", "content": "You are a precise data analyst."},
        {"role": "user", "content": prompt}
    ]
    return chat_completion(client, messages, model="gpt-4o-mini")



def _process_single_sheet(
    filename: str,
    sheet_name: str,
    raw_df: pd.DataFrame,
    alias_path: Optional[str],
    sheet_aliases: Optional[dict],
    client
) -> Tuple[pd.DataFrame, dict]:
    # 1) Clean / remap columns
    df = _clean_and_standardize_sheet(raw_df, alias_path)

    # 2) Normalize sheet type via alias map; then enforce heuristic if still unclassified
    try:
        norm_type = normalize_sheet_type(sheet_name, df, sheet_aliases) if sheet_aliases else "unclassified"
    except Exception:
        norm_type = "unclassified"

    if norm_type == "unclassified":
        heuristic = _detect_sheet_type_by_columns(df.columns, filename)
        if heuristic != "unclassified":
            norm_type = heuristic

    # OPTIONAL: preserve name vs feature hints (if helper exists)
    try:
        from sheet_utils import classify_sheet
        cls = classify_sheet(sheet_name, df, sheet_aliases)
        norm_type = cls.get("final_type", norm_type)
        name_hint = cls.get("name_hint", "unclassified")
        feature_hint = cls.get("feature_hint", "unclassified")
        type_resolution = cls.get("type_resolution", "unknown")
    except Exception:
        name_hint = "unclassified"
        feature_hint = "unclassified"
        type_resolution = "unknown"

    # 3) Add metadata columns (non-destructive)
    if "source_sheet" not in df.columns:
        df["source_sheet"] = sheet_name
    df["normalized_sheet_type"] = norm_type

    # 4) EDA (best-effort)
    try:
        eda_text = generate_eda_summary(df)
    except Exception as e_eda:
        eda_text = f"(EDA error: {type(e_eda).__name__}: {e_eda})"

    # 5) Summary (GPT if available, then fallback)
    summary_text = ""
    try:
        if client is not None and chat_completion is not None:
            summary_text = _gpt_summary_for_sheet(client, norm_type, filename, df, eda_text).strip()
    except Exception as e:
        summary_text = f"⚠️ GPT summary failed: {e}"

    if not summary_text:
        summary_text = _summarize_inventory_fallback(df) if norm_type == "inventory" \
            else f"{norm_type.title()} sheet. Rows={len(df)}. Cols={len(df.columns)}."

    meta = {
        "filename": os.path.basename(filename),
        "sheet_name": sheet_name,
        "normalized_sheet_type": norm_type,
        "name_implied_type": name_hint,
        "feature_implied_type": feature_hint,
        "type_resolution": type_resolution,
        "columns": list(map(str, df.columns)),
        "record_count": int(len(df)),
        "summary_text": summary_text
    }
    return df, meta
def _hardened_process_excel(
    xls: pd.ExcelFile,
    filename: Optional[str],
    alias_path: Optional[str],
    paths: Optional[Any]
) -> Tuple[Dict[str, pd.DataFrame], list, Optional[str]]:
    """
    Core, production-safe processing used by both run_pipeline() and run_pipeline_on_file().
    - Never drops data
    - Adds normalized_sheet_type when possible
    - Captures per-sheet errors in metadata
    - Returns (cleaned_sheets, per_sheet_meta, gpt_client_error)
    """
    cleaned_sheets: Dict[str, pd.DataFrame] = {}
    per_sheet_meta = []
    gpt_client_error: Optional[str] = None

    sheet_aliases = _resolve_sheet_aliases(paths)

    # Init GPT client safely (only once)
    client = None
    if get_openai_client is not None and chat_completion is not None:
        try:
            client = get_openai_client()
        except Exception as e:
            gpt_client_error = str(e)

    for sheet in xls.sheet_names:
        try:
            raw_df = pd.read_excel(xls, sheet_name=sheet)
            df, meta = _process_single_sheet(
                filename=filename or "(unknown)",
                sheet_name=sheet,
                raw_df=raw_df,
                alias_path=alias_path,
                sheet_aliases=sheet_aliases,
                client=client
            )
            cleaned_sheets[sheet] = df
            per_sheet_meta.append(meta)

        except Exception as e:
            per_sheet_meta.append({
                "filename": filename or "(unknown)",
                "sheet_name": sheet,
                "error": f"{type(e).__name__}: {e}",
            })

    return cleaned_sheets, per_sheet_meta, gpt_client_error


# ---------------- Legacy disk-writing entrypoint (kept, now hardened) ----------------

def run_pipeline_on_file(xls_path, alias_path, output_prefix, output_folder):
    xls = pd.ExcelFile(xls_path)
    cleaned_sheets_dict, per_sheet_meta, gpt_err = _hardened_process_excel(
        xls=xls,
        filename=os.path.basename(xls_path),
        alias_path=alias_path,
        paths=None  # legacy path variant doesn't pass full paths; that's fine
    )

    # Save metadata (both per-sheet and top-level)
    metadata = {
        "run_started": datetime.utcnow().isoformat() + "Z",
        "run_completed": datetime.utcnow().isoformat() + "Z",
        "source_filename": os.path.basename(xls_path),
        "sheet_count": len(xls.sheet_names),
        "processed_sheets": list(cleaned_sheets_dict.keys()),
        "sheets": per_sheet_meta,
    }
    if gpt_err:
        metadata["gpt_client_error"] = gpt_err

    # If your save_master_metadata_index expects a master structure, keep compatibility:
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

    cleaned_sheets, per_sheet_meta, gpt_err = _hardened_process_excel(
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
    if gpt_err:
        metadata["gpt_client_error"] = gpt_err

    return cleaned_sheets, metadata


