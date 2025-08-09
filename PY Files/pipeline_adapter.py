# PY Files/pipeline_adapter.py
import io
import os
import tempfile
from typing import Dict, Tuple

# Import your real pipeline module
from phase1_ingest import pipeline as pl

# Reuse your cloud helpers
from path_utils import get_project_paths  # not used here but OK to keep symmetry
from dbx_utils import read_file_bytes as dbx_read_bytes

def run_pipeline_cloud(filename: str, paths) -> Tuple[Dict[str, "pd.DataFrame"], dict]:
    """
    Adapter so main.py can call: cleaned_sheets, metadata = run_pipeline_cloud(filename, paths)
    - Downloads RAW/<filename> from Dropbox
    - Calls your underlying pipeline with either bytes or a temp local path (tries both)
    - Returns (cleaned_sheets: dict[str, DataFrame], metadata: dict)
    """
    # 1) Read source bytes from Dropbox RAW folder
    raw_path = f"{paths.raw_folder}/{filename}"
    file_bytes = dbx_read_bytes(raw_path)

    # 2) Try most common signatures FIRST
    #    A) run_pipeline(file_bytes: BytesIO, filename=..., paths=...)
    #    B) run_pipeline(file_path: str, filename=..., paths=...)
    #    C) run_pipeline(file_path: str)  or run_pipeline(file_bytes: BytesIO)
    # Normalize return to (cleaned_sheets, metadata)

    # Try bytes-based first
    try:
        bio = io.BytesIO(file_bytes)
        out = pl.run_pipeline(bio, filename=filename, paths=paths)  # type: ignore
        cleaned_sheets, metadata = _normalize_pipeline_output(out, filename)
        return cleaned_sheets, metadata
    except Exception:
        pass

    # Try local path with full args
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, filename)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        # B) full args
        try:
            out = pl.run_pipeline(tmp_path, filename=filename, paths=paths)  # type: ignore
            cleaned_sheets, metadata = _normalize_pipeline_output(out, filename)
            return cleaned_sheets, metadata
        except Exception:
            # C) minimal args
            out = pl.run_pipeline(tmp_path)  # type: ignore
            cleaned_sheets, metadata = _normalize_pipeline_output(out, filename)
            return cleaned_sheets, metadata


def _normalize_pipeline_output(out, filename: str):
    """
    Accepts several shapes:
      - (dict[str, DataFrame], dict)
      - {"sheets": dict[str, DataFrame], "metadata": dict}
      - {"cleaned_sheets": ..., "metadata": ...}
    Returns (cleaned_sheets, metadata) always.
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

    # Fallback minimal metadata if pipeline only wrote files
    return out, {"source_filename": filename, "note": "Metadata synthesized by adapter"}
