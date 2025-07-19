import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from io import BytesIO
import json

def get_drive_service():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gdrive_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=credentials)

def list_all_supported_files(folder_id):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, modifiedTime, size, parents)"
    ).execute()
    files = results.get('files', [])
    # Add a 'relative_path' key for compatibility
    for f in files:
        f["relative_path"] = f["name"]
    return files

def download_file(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def get_file_last_modified(file_id):
    service = get_drive_service()
    file = service.files().get(fileId=file_id, fields="modifiedTime").execute()
    return file.get("modifiedTime")

def load_json_file(file_id):
    fh = download_file(file_id)
    return json.load(fh)

def save_json_file(file_id, data):
    service = get_drive_service()
    media = MediaIoBaseUpload(BytesIO(json.dumps(data).encode()), mimetype='application/json')
    service.files().update(fileId=file_id, media_body=media).execute()