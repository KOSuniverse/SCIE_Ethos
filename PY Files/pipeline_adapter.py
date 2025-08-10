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
    Safe adapter — supports these call shapes:
      run_pipeline_cloud(filename, paths)
      run_pipeline_cloud(filename, paths, "dropbox")
      run_pipeline_cloud(filename, paths, storage="local")
      run_pipeline_cloud(filename, paths, file_bytes=b"...")
      (defensive) run_pipeline_cloud(filename, "dropbox", paths)  # swapped args
      (defensive) run_pipeline_cloud(filename, paths, paths)     # accidental dup
    """
    paths, storage_name = _normalize_args(paths, storage)

    # --- Acquire file bytes
    if file_bytes is None:
        if storage_name == "dropbox":
            raw_path = f"{_rstrip_slash(paths.raw_folder)}/{filename}"
            file_bytes = dbx_read_bytes(raw_path)
        elif storage_name == "local":
            local_path = os.path.join(paths.raw_folder, filename)
            with open(local_path, "rb") as f:
                file_bytes = f.read()
        else:
            # Unknown → fall back to Dropbox
            raw_path = f"{_rstrip_slash(paths.raw_folder)}/{filename}"
            file_bytes = dbx_read_bytes(raw_path)

    if not file_bytes:
        raise FileNotFoundError(f"Could not load bytes for '{filename}' (storage={storage_name}).")

    # --- Try BytesIO-based call first
    bio = io.BytesIO(file_bytes)
    try:
        out = pl.run_pipeline(bio, filename=filename, paths=paths)  # type: ignore
        return _normalize_pipeline_output(out, filename)
    except Exception:
        pass

    # --- Try temp file path variants
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


# ---------- helpers ----------

def _normalize_args(paths: Any, storage: Any) -> Tuple[Any, str]:
    """
    Returns (paths_obj, storage_name).
    - If args are swapped (paths is str and storage has .raw_folder), swap them.
    - If both look wrong, raise a clear error.
    """
    # Case 1: correct order already
    if hasattr(paths, "raw_folder") and not isinstance(storage, type(paths)):
        return paths, _normalize_storage(storage)

    # Case 2: swapped (paths is str-like, storage is paths object)
    if isinstance(paths, str) and hasattr(storage, "raw_folder"):
        return storage, _normalize_storage(paths)

    # Case 3: duplicate paths passed (paths, paths)
    if hasattr(paths, "raw_folder") and hasattr(storage, "raw_folder"):
        # Prefer the first, default storage to 'dropbox'
        return paths, "dropbox"

    # Case 4: both strings — assume first was meant to be storage (rare)
    if isinstance(paths, str) and isinstance(storage, str):
        raise TypeError(
            "run_pipeline_cloud expected a paths object, but received two strings. "
            "Call as run_pipeline_cloud(filename, paths, storage='dropbox')."
        )

    # Otherwise ambiguous
    raise TypeError(
        "run_pipeline_cloud could not identify the 'paths' object. "
        "Pass a valid paths object with a 'raw_folder' attribute."
    )


def _normalize_storage(storage: Any) -> str:
    """Return a safe storage name ('dropbox' or 'local'), defaulting to 'dropbox'."""
    if isinstance(storage, str):
        s = storage.strip().lower()
        return s if s in {"dropbox", "local"} else "dropbox"
    # If caller passed a paths-like object here, default to dropbox
    if hasattr(storage, "raw_folder"):
        return "dropbox"
    return "dropbox"


def _rstrip_slash(p: str) -> str:
    return p[:-1] if p.endswith("/") else p


def _normalize_pipeline_output(out: Any, filename: str) -> Tuple[Dict[str, "pd.DataFrame"], dict]:
    """
    Accepts several shapes:
      - (dict[str, DataFrame], dict)
      - {"sheets": dict[str, DataFrame], "metadata": dict}
      - {"cleaned_sheets": ..., "metadata": ...}
      - dict[str, DataFrame]  (no metadata)
    Returns (cleaned_sheets, metadata).
    """
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]

    if isinstance(out, dict):
        if "cleaned_sheets" in out and "metadata" in out:
            return out["cleaned_sheets"], out["metadata"]
        if "sheets" in out and "metadata" in out:
            return out["sheets"], out["metadata"]
        if out and all(isinstance(k, str) for k in out.keys()):
            return out, {"source_filename": filename, "note": "Metadata synthesized by adapter"}

    return out, {"source_filename": filename, "note": "Metadata synthesized by adapter"}



