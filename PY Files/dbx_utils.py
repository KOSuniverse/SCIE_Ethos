# PY Files/dbx_utils.py
import io
import os
from typing import List, Dict, Optional

try:
    import streamlit as st
except ImportError:
    st = None

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

def list_xlsx(folder_path: str) -> List[Dict]:
    """List .xlsx files in a Dropbox folder."""
    import dropbox
    dbx = _get_dbx_client()
    resp = dbx.files_list_folder(folder_path)
    files = []
    for e in resp.entries:
        if isinstance(e, dropbox.files.FileMetadata) and e.name.lower().endswith(".xlsx"):
            files.append({"name": e.name, "path_lower": e.path_lower})
    return files

def read_file_bytes(path_lower: str) -> bytes:
    """Read a file from Dropbox into memory."""
    dbx = _get_dbx_client()
    md, res = dbx.files_download(path_lower)
    return res.content
