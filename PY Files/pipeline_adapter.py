# PY Files/pipeline_adapter.py
import io
import os
import tempfile
from typing import Dict, Tuple, Any, Optional

from phase1_ingest import pipeline as pl
from dbx_utils import read_file_bytes as dbx_read_bytes

BytesLike = (bytes, bytearray, io.BytesIO)

def run_pipeline_cloud(
    source: Any,
    filename: str,
    paths: Any,
    storage: Optional[str] = None,
    *,
    file_bytes: Optional[bytes] = None
) -> Tuple[Dict[str, "pd.DataFrame"], dict]:
    """
    Final adapter matching main.py usage:
      cleaned_sheets, metadata = run_pipeline_cloud(b, filename, app_paths)

    Accepted 'source':
      - bytes/bytearray/BytesIO  -> use directly
      - str Dropbox path         -> read via dbx
      - str local path           -> open directly
      - None                     -> load by (storage, filename, paths)

    Behavior:
      - Prefer Dropbox folders when paths.dbx_* are present.
      - Never assumes paths.raw_folder exists; checks before using.
      - Always returns (cleaned_sheets: dict[str, DataFrame], metadata: dict).
    """
    # 0) Normalize storage hint
    storage_name = _normalize_storage(storage, paths)

    # 1) Acquire bytes or path handle from 'source'
    if isinstance(source, BytesLike):
        b = _to_bytes(source)
    elif isinstance(source, str) and source.strip():
        # A string source can be a Dropbox path or a local path
        if source.startswith("/"):
            b = dbx_read_bytes(source)
        else:
            with open(source, "rb") as f:
                b = f.read()
    elif source is None:
        # Fallback: locate by filename using storage + paths
        b = _load_by_filename(filename, paths, storage_name)
    else:
        raise TypeError(f"Unsupported 'source' type: {type(source).__name__}")

    if not b:
        raise FileNotFoundError(f"Could not load bytes for '{filename}' (storage={storage_name}).")

    # 2) Call your real pipeline with bytes first, then temp path variants
    bio = io.BytesIO(b)

    # A) BytesIO with explicit kwargs
    try:
        out = pl.run_pipeline(bio, filename=filename, paths=paths)  # type: ignore
        return _normalize_pipeline_output(out, filename)
    except Exception:
        pass

    # B/C) Temp file path attempts (with/without kwargs)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, filename)
        with open(tmp_path, "wb") as f:
            f.write(b)

        try:
            out = pl.run_pipeline(tmp_path, filename=filename, paths=paths)  # type: ignore
            return _normalize_pipeline_output(out, filename)
        except Exception:
            pass

        try:
            out = pl.run_pipeline(tmp_path)  # type: ignore
            return _normalize_pipeline_output(out, filename)
        except Exception:
            pass

    # D) Last resort: BytesIO only
    out = pl.run_pipeline(bio)  # type: ignore
    return _normalize_pipeline_output(out, filename)


# ---------------- helpers ----------------

def _to_bytes(x: Any) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, io.BytesIO):
        pos = x.tell()
        x.seek(0)
        b = x.read()
        x.seek(pos)
        return b
    raise TypeError("Not bytes-like")

def _normalize_storage(storage: Optional[str], paths: Any) -> str:
    """
    Decide storage mode. Prefer Dropbox if paths has dbx folders.
    """
    if isinstance(storage, str) and storage.strip().lower() in {"dropbox", "local"}:
        return storage.strip().lower()
    # If AppPaths exposes Dropbox folders, default to dropbox
    if hasattr(paths, "dbx_raw_folder") and getattr(paths, "dbx_raw_folder", None):
        return "dropbox"
    return "local"

def _load_by_filename(filename: str, paths: Any, storage_name: str) -> bytes:
    """
    Load bytes using filename + paths + storage hint, without assuming local raw_folder exists.
    """
    if storage_name == "dropbox":
        dbx_raw = getattr(paths, "dbx_raw_folder", None)
        if not dbx_raw:
            raise RuntimeError("Dropbox mode requested but 'dbx_raw_folder' is not set on paths.")
        dbx_path = f"{dbx_raw.rstrip('/')}/{filename}"
        return dbx_read_bytes(dbx_path)

    # local
    local_raw = getattr(paths, "raw_folder", None)
    if not local_raw:
        raise RuntimeError("Local mode requested but 'raw_folder' is not set on paths.")
    local_path = os.path.join(local_raw, filename)
    with open(local_path, "rb") as f:
        return f.read()

def _normalize_pipeline_output(out: Any, filename: str) -> Tuple[Dict[str, "pd.DataFrame"], dict]:
    # Tuple already
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    # Dict shapes
    if isinstance(out, dict):
        if "cleaned_sheets" in out and "metadata" in out:
            return out["cleaned_sheets"], out["metadata"]
        if "sheets" in out and "metadata" in out:
            return out["sheets"], out["metadata"]
        # Heuristic: dict[str, DataFrame]
        if out and all(isinstance(k, str) for k in out.keys()):
            return out, {"source_filename": filename, "note": "Metadata synthesized by adapter"}
    # Fallback: unknown shape
    return out, {"source_filename": filename, "note": "Metadata synthesized by adapter"}



