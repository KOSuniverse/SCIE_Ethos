# --- utils/gdrive.py ---

import streamlit as st
from gdrive_utils import (
    list_all_supported_files as _list_all_supported_files,
    download_file as _download_file,
    get_file_last_modified as _get_file_last_modified,
    load_json_file,
    save_json_file,
    get_drive_service
)

# Project constants
PROJECT_ROOT_ID = "1t1CcZzwsjOPMNKKMkdJd6kXhixTreNuY"

def get_gdrive_id_by_name(name, parent_id, is_folder=False):
    """Find a file or folder by name within a parent directory."""
    files = list_all_supported_files(parent_id)
    for f in files:
        if f["name"] == name:
            if is_folder and f["mimeType"] == "application/vnd.google-apps.folder":
                return f["id"]
            if not is_folder and f["mimeType"] != "application/vnd.google-apps.folder":
                return f["id"]
    return None

def list_all_supported_files(folder_id=None):
    """List all supported files, excluding raw data folder."""
    if folder_id is None:
        folder_id = PROJECT_ROOT_ID
    
    all_files = [
        f
        for f in _list_all_supported_files(folder_id)
        if not f["name"].startswith("Raw_Data")
    ]
    return all_files

def download_file(file_id):
    """Download a file from Google Drive."""
    return _download_file(file_id)

def get_file_last_modified(file_id):
    """Get the last modified time for a file."""
    return _get_file_last_modified(file_id)

# Initialize folder IDs
def get_models_folder_id():
    """Get the Models folder ID from 04_Data/Models."""
    data_folder_id = get_gdrive_id_by_name("04_Data", PROJECT_ROOT_ID, is_folder=True)
    if not data_folder_id:
        st.error("❌ Could not find 04_Data folder in Google Drive.")
        st.stop()
    
    models_folder_id = get_gdrive_id_by_name("Models", data_folder_id, is_folder=True)
    if not models_folder_id:
        st.error("❌ Could not find 04_Data/Models in Google Drive.")
        st.stop()
    
    return models_folder_id

def get_metadata_folder_id():
    """Get the metadata folder ID from 01_Project_Plan/_metadata."""
    project_plan_id = get_gdrive_id_by_name("01_Project_Plan", PROJECT_ROOT_ID, is_folder=True)
    if not project_plan_id:
        st.error("Could not find 01_Project_Plan in Project_Root.")
        st.stop()
    
    metadata_folder_id = get_gdrive_id_by_name("_metadata", project_plan_id, is_folder=True)
    if not metadata_folder_id:
        st.error("Could not find 01_Project_Plan/_metadata in Project_Root.")
        st.stop()
    
    return metadata_folder_id
