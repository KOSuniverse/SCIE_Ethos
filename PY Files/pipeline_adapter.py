# PY Files/pipeline_adapter.py
import io
import os
import tempfile
from typing import Dict, Tuple, Any, Optional

# Import your real pipeline module
from phase1_ingest import pipeline as pl

# Reuse your cloud/helpers
from path_utils import get_project_paths  # OK to keep for symmetry
from dbx_utils import read_file_bytes as dbx_read_bytes


def run_pipeline_cloud(
    filename: str,
    paths: Any,
    storage: Any = "dropbox",
    *,
    file_bytes: Optional[bytes] = None
) -> Tuple[Dict[str, "pd.DataFrame"], dict]:
    """
    Adapter so main.py can call in multiple ways without breaking:
      cleaned_sheets, metadata = run_pipeline_cloud(filename, paths)
      cleaned_sheets, metadata = run_pipeline_cloud(filename, paths, "dropbox")
      cleaned_sheets, metadata = run_pipeline_cloud(filename, paths, storage="local")
      cleaned_sheets, metadata = run_pipeline_cloud(filename, paths, file_bytes=b"...")
      (defensive) run_pipeline_cloud(filename, paths, paths)  # mistakenly passing paths twice

    Behavior
    - If file_bytes provided, use those.
    - Else if storage == "dropbox", download RAW/<filename> via dbx_read_bytes.
    - Else if storage == "local", read from os.path.join(paths.raw_folder, filename).
    - Calls pl.run_pipeline with best-guess signatures:
        A) run_pipeline(BytesIO, filename=..., paths=...)
        B) run_pipeline(file_path_str, filename=..., paths=...)
        C) run_pipeline(file_path_str)
        D) run_pipeline(BytesIO)
    - Always returns (cleaned_sheets: dict[str, DataFrame], metadata: dict)
    """

    # --- 0) Normalize storage in case the caller passed `paths` as the 3rd arg by mistake
    storage_name = _normalize_storage(storage)

    # --- 1) Acquire file bytes
    if file_bytes is None:
        if storage_name == "dropbox":
            # Ensure consistent path join for Dropbox
            raw_path = f"{paths.raw_folder.rstrip('/')}/{filename}"
            file_bytes = dbx_read_bytes(raw_path)
        elif storage_name == "local":
            local_path = os.path.join(paths.raw_folder, filename)
            with open(local_path, "rb") as f:
                file_bytes = f.read()
        else:
            # Unknown storage → fall back to Dropbox
            raw_path = f"{paths.raw_folder.rstrip('/')}/{filename}"
            file_bytes = dbx_read_bytes(raw_path)

    if not file_bytes:
        raise FileNotFoundError(f"Could not load bytes for '{filename}' (storage={storage_name}).")

    # --- 2) Try BytesIO-based call first
    bio = io.BytesIO(file_bytes)
    try:
        out = pl.run_pipeline(bio, filename=filename, paths=paths)  # type: ignore
        return _normalize_pipeline_output(out, filename)
    except Exception:
        pass

    # --- 3) Try temp file path variants
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, filename)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        # B) full args
        try:
            out = pl.run_pipeline(tmp_path, filename=filename, paths=paths)  # type: ignore
            return _normalize_pipeline_output(out, filename)
        except Exception:
            pass

        # C) minimal args (path only)
        try:
            out = pl.run_pipeline(tmp_path)  # type: ignore
            return _normalize_pipeline_output(out, filename)
        except Exception:
            pass

        # D) minimal args (BytesIO only)
        try:
            out = pl.run_pipeline(bio)  # type: ignore
            return _normalize_pipeline_output(out, filename)
        except Exception as e:
            raise RuntimeError(
                f"run_pipeline failed for '{filename}' using all known signatures."
            ) from e


def _normalize_storage(storage: Any) -> str:
    """
    Return a safe storage name.
    - Strings → lowercased ('dropbox'/'local')
    - If the caller accidentally passed a paths-like object (has .raw_folder), default to 'dropbox'
    - Anything else defaults to 'dropbox'
    """
    if isinstance(storage, str):
        return storage.lower().strip() or "dropbox"
    # Treat objects with raw_folder as 'paths' accidentally passed in
    if hasattr(storage, "raw_folder"):
        return "dropbox"
    return "dropbox"


def _normalize_pipeline_output(out: Any, filename: str) -> Tuple[Dict[str, "pd.DataFrame"], dict]:
    """
    Accepts several shapes:
      - (dict[str, DataFrame], dict)
      - {"sheets": dict[str, DataFrame], "metadata": dict}
      - {"cleaned_sheets": ..., "metadata": ...}
      - dict[str, DataFrame]  (no metadata)
    Returns (cleaned_sheets, metadata).
    """
    # Tuple already
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]

    # Dict shapes
    if isinstance(out, dict):
        if "cleaned_sheets" in out and "metadata" in out:
            return out["cleaned_sheets"], out["metadata"]
        if "sheets" in out and "metadata" in out:
            return out["sheets"], out["metadata"]

        # Heuristic: dictionary of DataFrames without metadata
        if out and all(isinstance(k, str) for k in out.keys()):
            return out, {"source_filename": filename, "note": "Metadata synthesized by adapter"}

    # Fallback: unknown shape
    return out, {"source_filename": filename, "note": "Metadata synthesized by adapter"}


