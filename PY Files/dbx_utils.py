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
            if name_lower.endswith(".xlsx"):
                files.append({
                    "name": e.name,
                    "path_lower": e.path_lower,
                    "server_modified": getattr(e, "server_modified", None),
                    "file_type": "excel"
                })

    files.sort(key=lambda d: d.get("server_modified") or 0, reverse=True)
    return files

def read_file_bytes(path_lower: str) -> bytes:
    """Read a file from Dropbox into memory."""
    print(f"DEBUG read_file_bytes: Attempting to read from Dropbox: {path_lower}")
    try:
        dbx = _get_dbx_client()
        md, res = dbx.files_download(path_lower)
        print(f"DEBUG read_file_bytes: Successfully downloaded {len(res.content)} bytes")
        return res.content
    except Exception as e:
        print(f"DEBUG read_file_bytes: Error downloading {path_lower}: {e}")
        raise

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


def append_jsonl_line(path_lower: str, record: dict) -> bool:
    """
    Append a single JSON record to a JSONL file in Dropbox.
    Used for logging events to Dropbox without filesystem operations.
    
    Args:
        path_lower: Dropbox path (e.g., "/Project_Root/04_Data/04_Metadata/app_events.jsonl")
        record: Dictionary to append as JSON line
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        dbx = _get_dbx_client()
        if not dbx:
            return False
            
        # Convert record to JSON line
        json_line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = json_line.encode('utf-8')
        
        # Try to append to existing file, or create new if doesn't exist
        try:
            # First try to read existing file
            existing_data = read_file_bytes(path_lower)
            if existing_data:
                # Append to existing content
                new_content = existing_data + line_bytes
            else:
                # File doesn't exist, start with just this line
                new_content = line_bytes
        except Exception:
            # File doesn't exist or can't be read, start fresh
            new_content = line_bytes
            
        # Upload the combined content
        upload_bytes(path_lower, new_content, mode="overwrite")
        return True
        
    except Exception as e:
        print(f"Failed to append JSONL line to {path_lower}: {e}")
        return False

# ---------------- OpenAI Files / Vector Store sync (cloud-only) ----------------

def _get_openai_client_safe():
    """Lazy import OpenAI client via llm_client, with a clear error if missing."""
    try:
        from llm_client import get_openai_client  # type: ignore
        return get_openai_client()
    except Exception as e:
        raise RuntimeError(f"OpenAI client not available: {e}")

def upload_dropbox_file_to_openai(path_lower: str, *, purpose: str = "assistants", filename: Optional[str] = None) -> Optional[str]:
    """Upload a Dropbox file to OpenAI Files without using local disk. Returns file_id or None."""
    try:
        content = read_file_bytes(path_lower)
        name = filename or os.path.basename(path_lower) or "file.bin"
        
        bio = io.BytesIO(content)
        # Some SDKs require a .name on the file-like object
        try:
            setattr(bio, "name", name)
        except Exception:
            pass
        client = _get_openai_client_safe()
        file_obj = client.files.create(file=bio, purpose=purpose)
        print(f"Uploaded {name} to OpenAI with file_id: {getattr(file_obj, 'id', None)}")
        return getattr(file_obj, "id", None)
    except Exception as e:
        print(f"OpenAI file upload failed for {path_lower}: {e}")
        return None

def upload_many_dropbox_files_to_openai(paths: List[str], *, purpose: str = "assistants") -> List[str]:
    """Upload many Dropbox files to OpenAI Files and return list of file IDs (skips failures)."""
    ids: List[str] = []
    for p in paths:
        fid = upload_dropbox_file_to_openai(p, purpose=purpose)
        if fid:
            ids.append(fid)
    return ids

def create_vector_store_with_files(name: str, file_ids: List[str]) -> Optional[str]:
    """Create an OpenAI vector store and batch-add the given file IDs. Returns vector_store_id or None."""
    if not file_ids:
        print("No file IDs provided to create_vector_store_with_files")
        return None
    try:
        client = _get_openai_client_safe()
        print(f"Creating vector store '{name}' with {len(file_ids)} files: {file_ids}")
        vs = client.beta.vector_stores.create(name=name)
        print(f"Vector store created: {vs.id}")
        
        # Wait for vector store to be ready and add files
        batch_result = client.beta.vector_stores.file_batches.create(
            vector_store_id=vs.id, 
            file_ids=file_ids
        )
        print(f"File batch created: {batch_result.id}")
        return vs.id
    except Exception as e:
        print(f"Vector store creation failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def prepare_file_search_from_dropbox(paths: List[str], *, vs_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience: upload Dropbox files to OpenAI and create a vector store for file_search.
    Returns {"file_ids": [...], "vector_store_id": "..."}.
    """
    print(f"prepare_file_search_from_dropbox called with {len(paths)} paths: {paths}")
    file_ids = upload_many_dropbox_files_to_openai(paths)
    print(f"File upload result: {len(file_ids)} files uploaded: {file_ids}")
    vs_id = None
    if file_ids:
        vs_id = create_vector_store_with_files(vs_name or "Ethos_FileStore", file_ids)
        print(f"Vector store creation result: {vs_id}")
    else:
        print("No files uploaded, skipping vector store creation")
    return {"file_ids": file_ids, "vector_store_id": vs_id}

def attach_vector_store_to_assistant(assistant_id: str, vector_store_id: str) -> bool:
    """Attach vector store to an existing Assistant for file_search tool."""
    try:
        client = _get_openai_client_safe()
        client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        return True
    except Exception as e:
        print(f"Assistant vector store attach failed: {e}")
        return False

def attach_vector_store_to_thread(thread_id: str, vector_store_id: str) -> bool:
    """Attach vector store to a thread for file_search in that conversation."""
    try:
        client = _get_openai_client_safe()
        client.beta.threads.update(
            thread_id=thread_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        return True
    except Exception as e:
        print(f"Thread vector store attach failed: {e}")
        return False
