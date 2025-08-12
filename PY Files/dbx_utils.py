# PY Files/dbx_utils.py
import io, json
import os
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import pandas as pd

# Enterprise foundation imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# ---------------- internal helpers ----------------

def _secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get secret from Streamlit secrets.toml or environment variables."""
    if STREAMLIT_AVAILABLE and hasattr(st, "secrets"):
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

def list_data_files(folder_path: str) -> List[Dict]:
    """List .xlsx and .csv files in a Dropbox folder (handles pagination)."""
    import dropbox
    dbx = _get_dbx_client()
    files: List[Dict] = []

    resp = dbx.files_list_folder(folder_path)
    entries = list(resp.entries)

    while resp.has_more:
        resp = dbx.files_list_folder_continue(resp.cursor)
        entries.extend(resp.entries)

    for e in entries:
        if isinstance(e, dropbox.files.FileMetadata):
            name_lower = e.name.lower()
            if name_lower.endswith((".xlsx", ".csv")):
                files.append({
                    "name": e.name,
                    "path_lower": e.path_lower,
                    "server_modified": getattr(e, "server_modified", None),
                    "file_type": "excel" if name_lower.endswith(".xlsx") else "csv"
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

def save_xlsx_bytes(
    source: Union[Dict[str, pd.DataFrame], bytes, bytearray, io.BytesIO]
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


def upload_manifest(manifest_data: Dict[str, Any], manifest_path: str = None) -> str:
    """
    Upload a manifest file to Dropbox with enterprise metadata.
    
    Args:
        manifest_data: Manifest dictionary to upload
        manifest_path: Optional Dropbox path. If None, generates timestamped path.
        
    Returns:
        str: Dropbox path where manifest was uploaded
    """
    if manifest_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = f"/manifests/sync_manifest_{timestamp}.json"
    
    # Enhance manifest with Dropbox-specific metadata
    enhanced_manifest = manifest_data.copy()
    enhanced_manifest["cloud_sync"] = enhanced_manifest.get("cloud_sync", {})
    enhanced_manifest["cloud_sync"].update({
        "dropbox_path": manifest_path,
        "dropbox_root": _secret("DROPBOX_ROOT", ""),
        "upload_timestamp": datetime.now().isoformat()
    })
    
    upload_json(manifest_path, enhanced_manifest)
    return manifest_path


def list_manifests(folder_path: str = "/manifests") -> List[Dict[str, Any]]:
    """
    List all manifest files in a Dropbox folder.
    
    Args:
        folder_path: Dropbox folder to search for manifests
        
    Returns:
        List of manifest metadata dictionaries
    """
    import dropbox
    
    try:
        dbx = _get_dbx_client()
        manifests = []
        
        try:
            resp = dbx.files_list_folder(folder_path)
            entries = list(resp.entries)
            
            while resp.has_more:
                resp = dbx.files_list_folder_continue(resp.cursor)
                entries.extend(resp.entries)
            
            for entry in entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.json'):
                    manifests.append({
                        "name": entry.name,
                        "path": entry.path_lower,
                        "size": entry.size,
                        "server_modified": entry.server_modified,
                        "content_hash": getattr(entry, "content_hash", None)
                    })
            
            # Sort by modification date, newest first
            manifests.sort(key=lambda x: x.get("server_modified", datetime.min), reverse=True)
            
        except dropbox.exceptions.ApiError as e:
            if e.error.is_path_not_found():
                # Folder doesn't exist yet
                return []
            else:
                raise
                
    except Exception as e:
        print(f"Warning: Could not list Dropbox manifests: {e}")
        return []
    
    return manifests


def validate_dropbox_config() -> Dict[str, Any]:
    """
    Validate Dropbox configuration and connectivity.
    
    Returns:
        Dict with validation results
    """
    validation = {
        "config_valid": False,
        "connectivity": False,
        "errors": [],
        "config_source": "secrets.toml" if STREAMLIT_AVAILABLE else "environment"
    }
    
    # Check configuration completeness
    app_key = _secret("DROPBOX_APP_KEY")
    app_secret = _secret("DROPBOX_APP_SECRET") 
    refresh_token = _secret("DROPBOX_REFRESH_TOKEN")
    
    missing_config = []
    if not app_key:
        missing_config.append("DROPBOX_APP_KEY")
    if not app_secret:
        missing_config.append("DROPBOX_APP_SECRET")
    if not refresh_token:
        missing_config.append("DROPBOX_REFRESH_TOKEN")
    
    if missing_config:
        validation["errors"].append(f"Missing configuration: {', '.join(missing_config)}")
        return validation
    
    validation["config_valid"] = True
    
    # Test connectivity
    try:
        dbx = _get_dbx_client()
        
        # Test basic connectivity with account info
        account_info = dbx.users_get_current_account()
        validation["connectivity"] = True
        validation["account_info"] = {
            "name": account_info.name.display_name,
            "email": account_info.email,
            "account_id": account_info.account_id
        }
        
    except Exception as e:
        validation["errors"].append(f"Dropbox connection error: {str(e)}")
    
    return validation

