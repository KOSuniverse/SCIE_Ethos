# PY Files/tools_runtime.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import os
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

def _csv_bytes_to_frame(csv_bytes: bytes, filename: str) -> Dict[str, pd.DataFrame]:
    """Load a CSV byte blob into a DataFrame with the filename as sheet name."""
    try:
        # Try common encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                csv_text = csv_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # If all encodings fail, use utf-8 with error handling
            csv_text = csv_bytes.decode('utf-8', errors='replace')
        
        df = pd.read_csv(io.StringIO(csv_text))
        # Use filename without extension as sheet name
        sheet_name = os.path.splitext(filename)[0] if filename else "Sheet1"
        return {sheet_name: df}
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {"Sheet1": pd.DataFrame()}

def _load_file_to_frames(file_path: str) -> Dict[str, pd.DataFrame]:
    """Load either Excel or CSV file from Dropbox into DataFrame(s)."""
    print(f"DEBUG _load_file_to_frames: Attempting to load: {file_path}")
    
    try:
        file_bytes = read_file_bytes(file_path)
        print(f"DEBUG _load_file_to_frames: Successfully read {len(file_bytes)} bytes")
        filename = file_path.split('/')[-1]
        
        if filename.lower().endswith(('.xlsx', '.xls')):
            return _excel_bytes_to_frames(file_bytes)
        elif filename.lower().endswith('.csv'):
            return _csv_bytes_to_frame(file_bytes, filename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    except Exception as e:
        print(f"DEBUG _load_file_to_frames: Error loading {file_path}: {e}")
        raise

def _excel_bytes_to_frames_legacy(xlsx_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Legacy function name for backwards compatibility."""
    return _excel_bytes_to_frames(xlsx_bytes)

def _save_csv_to_dropbox(df: pd.DataFrame, dbx_folder: str, base: str) -> str:
    """Save DataFrame as CSV to Dropbox and return the path."""
    path = f"{dbx_folder.rstrip('/')}/{base}.csv"
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    upload_bytes(path, bio.getvalue())
    return path

def _save_excel_to_dropbox(df: pd.DataFrame, dbx_folder: str, base: str, sheet_name: str = "Analysis") -> str:
    """Save DataFrame as Excel to Dropbox and return the path."""
    path = f"{dbx_folder.rstrip('/')}/{base}.xlsx"
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    bio.seek(0)
    upload_bytes(path, bio.getvalue())
    return path

def _save_artifact(df: pd.DataFrame, dbx_folder: str, base: str, format: str = "csv") -> str:
    """
    Save DataFrame artifact in the specified format.
    
    Args:
        df: DataFrame to save
        dbx_folder: Target Dropbox folder
        base: Base filename (without extension)
        format: "csv" or "excel"
    
    Returns:
        Path to saved file
    """
    if format.lower() == "excel":
        return _save_excel_to_dropbox(df, dbx_folder, base)
    else:
        return _save_csv_to_dropbox(df, dbx_folder, base)

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
            "purpose": "Filter/group/aggregate/join cleansed sheets from Excel or CSV files; save artifact.",
            "args_schema": {
                "cleansed_paths": ["dropbox://04_Data/01_Cleansed_Files/…xlsx or …csv"],
                "sheet": "sheet name within workbook (optional for Excel; ignored for CSV)",
                "filters": [{"col": "string", "op": "==|!=|>|>=|<|<=|in|contains", "value": "any"}],
                "groupby": ["col1","col2"],
                "metrics": [{"col": "extended_cost", "agg": "sum|min|max|mean|count"}],
                "limit": 50,
                "artifact_folder": "dropbox://04_Data/03_Summaries",
                "artifact_format": "csv (fast) or excel (OpenAI compatible)"
            },
            "returns": "dict with 'preview' (rows), 'columns', 'artifact_path'."
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
        },
        "list_sheets": {
            "purpose": "List sheet names in a cleansed workbook (Excel) or return filename for CSV from Dropbox.",
            "args_schema": {"cleansed_path": "dropbox://04_Data/01_Cleansed_Files/…xlsx or …csv"},
            "returns": "list of sheet names (or [filename] for CSV)"
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
    cleansed_paths: Optional[List[str]] = None,
    *,
    files: Optional[List[str]] = None,
    sheet: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    groupby: Optional[List[str]] = None,
    metrics: Optional[List[Dict[str, str]]] = None,
    limit: int = 50,
    artifact_folder: Optional[str] = None,
    artifact_format: str = "excel",  # "csv" or "excel" - excel for OpenAI compatibility
    **kwargs  # Accept additional arguments for compatibility
) -> Dict[str, Any]:
    """
    Enhanced dataframe_query shim (accepts legacy arg or kwargs) for backward compatibility.
    
    This function maintains compatibility with existing code while supporting new parameters.
    Delegates to phase2_analysis.dataframe_query for enterprise functionality.
    """
    
    # Handle parameter compatibility: files takes precedence, fallback to cleansed_paths
    paths = files if files is not None else cleansed_paths
    if not paths:
        return {"error": "No cleansed paths or files provided."}

    # Try to use enhanced dataframe_query if available
    try:
        from phase2_analysis.dataframe_query import dataframe_query as enhanced_query
        
        # Map parameters for enhanced function
        enhanced_kwargs = {
            "files": paths,
            "sheet": sheet,
            "filters": filters,
            "groupby": groupby,
            "metrics": metrics,
            "limit": limit,
            "artifact_folder": artifact_folder,
            "artifact_format": artifact_format
        }
        
        # Add any additional kwargs
        enhanced_kwargs.update(kwargs)
        
        return enhanced_query(**enhanced_kwargs)
        
    except ImportError:
        # Fallback to original implementation if enhanced version not available
        pass

    first = paths[0]
    
    try:
        frames = _load_file_to_frames(first)
    except Exception as e:
        return {"error": f"Failed to load file {first}: {str(e)}"}

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
        artifact_path = _save_artifact(df_out, artifact_folder, base, artifact_format)

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

def list_sheets(cleansed_path: str) -> List[str]:
    """
    Enhanced list_sheets function - List sheet names in a cleansed workbook (Excel) or return filename for CSV from Dropbox.
    
    Now exposed as a primary function per ChatGPT enterprise plan.
    """
    try:
        filename = cleansed_path.split('/')[-1].lower()
        if filename.endswith('.csv'):
            # For CSV files, return the filename without extension as the "sheet" name
            sheet_name = os.path.splitext(cleansed_path.split('/')[-1])[0]
            return [sheet_name]
        else:
            # For Excel files, list actual sheets
            frames = _load_file_to_frames(cleansed_path)
            return list(frames.keys())
    except Exception as e:
        print(f"Error listing sheets for {cleansed_path}: {e}")
        return []

def get_sheet_schema(cleansed_path: str, sheet_name: str = None) -> Dict[str, Any]:
    """
    NEW: Get schema information for a specific sheet.
    Useful for choose_aggregation() function in assistant_bridge.
    """
    try:
        frames = _load_file_to_frames(cleansed_path)
        
        # Choose sheet
        if sheet_name and sheet_name in frames:
            df = frames[sheet_name]
        else:
            # Use first sheet
            df = next(iter(frames.values()))
        
        # Build schema
        schema = {}
        for col in df.columns:
            schema[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist()
            }
        
        return {
            "columns": list(df.columns),
            "schema": schema,
            "row_count": len(df),
            "sheet_name": sheet_name or "(first sheet)"
        }
        
    except Exception as e:
        return {"error": f"Failed to get schema: {str(e)}"}

def get_sample_data(cleansed_path: str, sheet_name: str = None, limit: int = 5) -> List[Dict]:
    """
    NEW: Get sample data rows for assistant analysis.
    Used by choose_aggregation() to understand data structure.
    """
    try:
        frames = _load_file_to_frames(cleansed_path)
        
        # Choose sheet
        if sheet_name and sheet_name in frames:
            df = frames[sheet_name]
        else:
            # Use first sheet
            df = next(iter(frames.values()))
        
        # Return sample as list of dictionaries
        sample = df.head(limit).to_dict(orient='records')
        return sample
        
    except Exception as e:
        return [{"error": f"Failed to get sample data: {str(e)}"}]

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
