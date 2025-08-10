# PY Files/tools_runtime.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import json
import pandas as pd
import numpy as np

from dbx_utils import read_file_bytes, upload_bytes
# NOTE: Always use Dropbox folders from AppPaths (cloud-only)

# ---------- Helpers ----------

def _excel_bytes_to_frames(xlsx_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Load an Excel .xlsx byte blob into {sheet_name: DataFrame}."""
    with pd.ExcelFile(io.BytesIO(xlsx_bytes)) as xls:
        frames = {sn: xls.parse(sn) for sn in xls.sheet_names}
    return frames

def _save_csv_to_dropbox(df: pd.DataFrame, dbx_folder: str, base: str) -> str:
    """Save DataFrame as CSV to Dropbox and return the path."""
    path = f"{dbx_folder.rstrip('/')}/{base}.csv"
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    upload_bytes(path, bio.getvalue())
    return path

def _save_png_to_dropbox(fig, dbx_folder: str, base: str) -> str:
    """Save a Matplotlib figure to Dropbox as PNG and return the path."""
    path = f"{dbx_folder.rstrip('/')}/{base}.png"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    upload_bytes(path, buf.read())
    return path

# ---------- Public: Tool Catalog (descriptions for prompts) ----------

def tool_specs() -> Dict[str, Dict[str, Any]]:
    """Schema/metadata for the tool guide in prompts."""
    return {
        "cleanse_tool": {
            "purpose": "Run the cleansing pipeline on a file (already handled in UI; rarely needed in Q&A).",
            "args_schema": {"dropbox_path": "path to RAW file (not typically used in Q&A)"},
            "returns": "dict with 'cleansed_path', 'run_meta_path', 'sheets' (names only)."
        },
        "dataframe_query": {
            "purpose": "Filter/group/aggregate/join cleansed sheets; save CSV artifact.",
            "args_schema": {
                "cleansed_paths": ["dropbox://04_Data/01_Cleansed_Files/…xlsx"],
                "sheet": "sheet name within workbook (optional; default first sheet)",
                "filters": [{"col": "string", "op": "==|!=|>|>=|<|<=|in|contains", "value": "any"}],
                "groupby": ["col1","col2"],
                "metrics": [{"col": "extended_cost", "agg": "sum|min|max|mean|count"}],
                "limit": 50,
                "artifact_folder": "dropbox://04_Data/03_Summaries"
            },
            "returns": "dict with 'preview' (rows), 'columns', 'artifact_path' (CSV)."
        },
        "chart": {
            "purpose": "Create bar/line chart from rows and save PNG to Dropbox.",
            "args_schema": {
                "kind": "bar|line",
                "rows": [{"col": "val"}],
                "x": "column",
                "y": ["columns"],
                "title": "string",
                "artifact_folder": "dropbox://04_Data/02_EDA_Charts",
                "base_name": "filename base for the PNG"
            },
            "returns": "dict with 'image_path' and 'details'."
        },
        "kb_search": {
            "purpose": "Stub for knowledgebase; returns empty until Stage 6 is wired.",
            "args_schema": {"query": "string", "k": 5},
            "returns": "dict with 'chunks': [], 'citations': []"
        }
    }

# ---------- Tools ----------

def cleanse_tool(dropbox_path: str, app_paths: Any) -> Dict[str, Any]:
    """
    Typically not used from Q&A (the UI already runs cleanse).
    Provided for completeness if GPT needs to trigger a cleanse.
    """
    from pipeline_adapter import run_pipeline_cloud  # local import to avoid cycles
    filename = dropbox_path.rsplit("/", 1)[-1]
    cleaned_sheets, metadata = run_pipeline_cloud(filename, app_paths, storage="dropbox")
    # Save cleansed workbook bytes using existing dbx_utils flow (done in UI normally)
    # Here we just return sheet names and rely on UI to persist artifacts.
    return {
        "sheets": list(cleaned_sheets.keys()),
        "metadata": metadata,
    }

def _apply_filters(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not filters:
        return df
    out = df
    for f in filters:
        col = f.get("col")
        op  = (f.get("op") or "==").lower()
        val = f.get("value")
        if col not in out.columns:
            continue
        series = out[col]
        if op == "==":
            out = out[series == val]
        elif op == "!=":
            out = out[series != val]
        elif op == ">":
            out = out[pd.to_numeric(series, errors="coerce") > _to_num(val)]
        elif op == ">=":
            out = out[pd.to_numeric(series, errors="coerce") >= _to_num(val)]
        elif op == "<":
            out = out[pd.to_numeric(series, errors="coerce") < _to_num(val)]
        elif op == "<=":
            out = out[pd.to_numeric(series, errors="coerce") <= _to_num(val)]
        elif op == "in":
            vv = set(val if isinstance(val, list) else [val])
            out = out[series.astype(str).isin({str(x) for x in vv})]
        elif op == "contains":
            out = out[series.astype(str).str.contains(str(val), case=False, na=False)]
    return out

def _to_num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def dataframe_query(
    cleansed_paths: List[str],
    *,
    sheet: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    groupby: Optional[List[str]] = None,
    metrics: Optional[List[Dict[str, str]]] = None,
    limit: int = 50,
    artifact_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load the first cleansed workbook from Dropbox, run ops, save CSV, return preview.
    """
    if not cleansed_paths:
        return {"error": "No cleansed paths provided."}

    first = cleansed_paths[0]
    b = read_file_bytes(first)
    frames = _excel_bytes_to_frames(b)

    # Choose sheet
    if sheet and sheet in frames:
        df = frames[sheet].copy()
    else:
        # prefer a typed sheet if present
        df = None
        for sn, sdf in frames.items():
            if "normalized_sheet_type" in sdf.columns:
                df = sdf.copy()
                break
        if df is None:
            # fallback to first sheet
            first_name = next(iter(frames))
            df = frames[first_name].copy()

    # Filters
    df = _apply_filters(df, filters)

    # Groupby + metrics
    if groupby and metrics:
        agg_map = {}
        for m in metrics:
            col = m.get("col")
            agg = (m.get("agg") or "sum").lower()
            if col:
                agg_map[col] = agg
        grouped = df.groupby(groupby, dropna=False).agg(agg_map).reset_index()
        df_out = grouped
    elif metrics and not groupby:
        # simple aggregations over whole table
        rows = {f"{m.get('agg','sum')}({m.get('col')})": pd.to_numeric(df[m.get("col")], errors="coerce").agg(m.get("agg","sum"))
                for m in metrics if m.get("col") in df.columns}
        df_out = pd.DataFrame([rows])
    else:
        df_out = df

    # Artifact save
    artifact_path = None
    if artifact_folder:
        base = "df_query_result"
        artifact_path = _save_csv_to_dropbox(df_out, artifact_folder, base)

    # Preview
    prev = df_out.head(limit)
    return {
        "columns": list(map(str, prev.columns)),
        "preview": json.loads(prev.to_json(orient="records")),
        "artifact_path": artifact_path,
        "rowcount": int(len(df_out)),
        "sheet_used": sheet or "(auto)",
        "note": "All operations executed in cloud against Dropbox-hosted cleansed workbook."
    }

def chart(
    *,
    kind: str,
    rows: List[Dict[str, Any]],
    x: str,
    y: List[str],
    title: Optional[str],
    artifact_folder: str,
    base_name: str = "chart"
) -> Dict[str, Any]:
    """
    Create a simple bar/line chart from provided rows; save PNG to Dropbox.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    if x not in df.columns or any(col not in df.columns for col in y):
        return {"error": "x or y columns missing from rows."}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if kind == "line":
        for col in y:
            ax.plot(df[x], df[col], label=col)
    else:  # default bar
        width = 0.8 / max(1, len(y))
        xs = np.arange(len(df[x]))
        for i, col in enumerate(y):
            ax.bar(xs + i * width, df[col], width=width, label=col)
        ax.set_xticks(xs + width * (len(y)-1) / 2)
        ax.set_xticklabels(df[x].astype(str), rotation=0)

    ax.set_title(title or "")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    image_path = _save_png_to_dropbox(fig, artifact_folder, base_name)
    plt.close(fig)

    return {
        "image_path": image_path,
        "details": {"kind": kind, "x": x, "y": y, "title": title or ""},
    }

def kb_search(query: str, k: int = 5) -> Dict[str, Any]:
    """Stub until Stage 6 (OpenAI File Search)."""
    return {"chunks": [], "citations": [], "k": k, "query": query}
