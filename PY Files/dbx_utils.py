# PY Files/dbx_utils.py
import io, json
import os
from typing import List, Dict, Optional, Any, Union
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None

# ---------------- internal helpers ----------------

def _secret(name: str, default: Optional[str] = None) -> Optional[str]:
    if st is not None and hasattr(st, "secrets"):
        v = st.secrets.get(name)
        if v is not None:
            return v
    return os.getenv(name, default)

def _get_dbx_client():
    import dropbox
    app_key = _secret("DROPBOX_APP_KEY")
    app_secret = _secret("DROPBOX_APP_SECRET")
    refresh_token = _secret("DROPBOX_REFRESH_TOKEN")
    if not (app_key and app_secret and refresh_token):
        raise RuntimeError("Missing Dropbox secrets.")
    return dropbox.Dropbox(
        oauth2_refresh_token=refresh_token,
        app_key=app_key,
        app_secret=app_secret,
    )

# ---------------- listing / io ----------------

def list_xlsx(folder_path: str) -> List[Dict]:
    """List .xlsx files in a Dropbox folder (handles pagination)."""
    import dropbox
    dbx = _get_dbx_client()
    files: List[Dict] = []

    resp = dbx.files_list_folder(folder_path)
    entries = list(resp.entries)

    while resp.has_more:
        resp = dbx.files_list_folder_continue(resp.cursor)
        entries.extend(resp.entries)

    for e in entries:
        if isinstance(e, dropbox.files.FileMetadata) and e.name.lower().endswith(".xlsx"):
            files.append({
                "name": e.name,
                "path_lower": e.path_lower,
                "server_modified": getattr(e, "server_modified", None)
            })

    files.sort(key=lambda d: d.get("server_modified") or 0, reverse=True)
    return files

def read_file_bytes(path_lower: str) -> bytes:
    """Read a file from Dropbox into memory."""
    dbx = _get_dbx_client()
    md, res = dbx.files_download(path_lower)
    return res.content

def upload_bytes(path_lower: str, data: bytes, mode: str = "overwrite"):
    """Upload arbitrary bytes to Dropbox at path_lower."""
    import dropbox
    dbx = _get_dbx_client()
    write_mode = dropbox.files.WriteMode.overwrite if mode == "overwrite" else dropbox.files.WriteMode.add
    dbx.files_upload(data, path_lower, mode=write_mode, mute=True)

# ---------------- Excel bytes builder (hardened) ----------------

BytesLike = (bytes, bytearray, io.BytesIO)

def save_xlsx_bytes(
    source: Union[Dict[str, pd.DataFrame], BytesLike]
) -> bytes:
    """
    Build a valid .xlsx byte stream.

    Accepts:
      - dict[str, pd.DataFrame]  -> builds an Excel workbook with one sheet per key
      - bytes/bytearray/BytesIO  -> pass-through (already .xlsx content)

    Returns:
      - bytes: Excel file content ready for upload_bytes(...)

    Notes:
      - No filename/path is required.
      - Never calls .endswith(...) on anything.
      - Sheet names are truncated to 31 chars (Excel limit).
    """
    # Pass-through for already bytes-like content
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, io.BytesIO):
        pos = source.tell()
        source.seek(0)
        data = source.read()
        source.seek(pos)
        return data

    # Build from dict-of-DataFrames
    if isinstance(source, dict):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            for sheet_name, df in source.items():
                safe_name = (str(sheet_name) or "Sheet1")[:31]
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df.to_excel(writer, sheet_name=safe_name, index=False)
        buf.seek(0)
        return buf.read()

    raise TypeError(
        "save_xlsx_bytes() expected a dict[str, DataFrame] or bytes-like input."
    )

# ---------------- JSON upload ----------------

def upload_json(path_lower: str, obj: dict, mode: str = "overwrite"):
    """Upload a JSON object to Dropbox."""
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes(path_lower, data, mode=mode)
