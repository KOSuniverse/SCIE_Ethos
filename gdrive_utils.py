import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from io import BytesIO
import json
from dateutil.parser import isoparse

def get_drive_service():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gdrive_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=credentials)

def get_gdrive_id_by_name(name, parent_id, is_folder=False):
    service = get_drive_service()
    query = f"'{parent_id}' in parents and trashed = false and name = '{name}'"
    if is_folder:
        query += " and mimeType = 'application/vnd.google-apps.folder'"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)"
    ).execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

def list_all_supported_files(folder_id):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, modifiedTime, size, parents)"
    ).execute()
    files = results.get('files', [])
    for f in files:
        f["relative_path"] = f["name"]
    return files

def download_file(file_id):
    service = get_drive_service()
    file_meta = service.files().get(fileId=file_id, fields="mimeType, name").execute()
    if file_meta["mimeType"].startswith("application/vnd.google-apps."):
        raise Exception(f"⚠️ Cannot download native Google file (Docs, Sheets): {file_meta['name']}")
    request = service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def get_file_last_modified(file_id):
    service = get_drive_service()
    file = service.files().get(fileId=file_id, fields="modifiedTime").execute()
    ts = file.get("modifiedTime")
    return isoparse(ts).timestamp() if ts else None

def load_json_file(file_id):
    fh = download_file(file_id)
    return json.load(fh)

def save_json_file(file_id_or_none, data, filename=None, parent_folder_id=None):
    service = get_drive_service()
    buffer = BytesIO(json.dumps(data, indent=2).encode("utf-8"))
    buffer.seek(0)
    media = MediaIoBaseUpload(buffer, mimetype="application/json", resumable=True)

    if file_id_or_none:
        service.files().update(fileId=file_id_or_none, media_body=media).execute()
    elif filename and parent_folder_id:
        metadata = {
            "name": filename,
            "parents": [parent_folder_id],
            "mimeType": "application/json"
        }
        service.files().create(body=metadata, media_body=media).execute()
    else:
        raise ValueError("Missing file_id or creation metadata")

def save_model_file(model_bytes, model_name, models_folder_id):
    service = get_drive_service()
    buffer = BytesIO(model_bytes)
    buffer.seek(0)
    media = MediaIoBaseUpload(buffer, mimetype="application/octet-stream", resumable=True)
    metadata = {
        "name": model_name,
        "parents": [models_folder_id]
    }
    service.files().create(body=metadata, media_body=media).execute()
