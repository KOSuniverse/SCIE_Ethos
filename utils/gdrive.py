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
PROJECT_ROOT_ID = "1rhT8zhCu4FGy1YUxXABXhUYV92iZwlAE"  # Project_Root folder in Shared Drive

def get_gdrive_id_by_name(name, parent_id, is_folder=False):
    """Find a file or folder by name within a parent directory."""
    # Use the non-recursive function to avoid circular calls
    files = _list_all_supported_files(parent_id)
    for f in files:
        if f["name"] == name:
            if is_folder and f["mimeType"] == "application/vnd.google-apps.folder":
                return f["id"]
            if not is_folder and f["mimeType"] != "application/vnd.google-apps.folder":
                return f["id"]
    return None

def list_all_supported_files(folder_id=None):
    """List all supported files recursively, excluding raw data folder."""
    if folder_id is None:
        folder_id = PROJECT_ROOT_ID
    
    all_files = []
    
    def scan_folder_recursively(current_folder_id, path=""):
        """Recursively scan folders for supported files."""
        files_in_folder = _list_all_supported_files(current_folder_id)
        
        # 🔍 DEBUG LOG - Show what we found
        st.write(f"📂 Scanning: {path or 'ROOT'} ({current_folder_id}) — found {len(files_in_folder)} items")
        
        for f in files_in_folder:
            file_name = f["name"]
            mime_type = f.get("mimeType", "")
            
            # 🔍 DEBUG LOG - Show each item
            st.write(f"   🔎 {file_name} ({mime_type})")
            
            # Skip Raw_Data folders
            if file_name.startswith("Raw_Data"):
                st.write(f"   ⏭️ Skipping Raw_Data folder: {file_name}")
                continue
                
            if mime_type == "application/vnd.google-apps.folder":
                # It's a folder - scan it recursively
                new_path = f"{path}/{file_name}" if path else file_name
                st.write(f"   📁 Entering folder: {file_name}")
                scan_folder_recursively(f["id"], new_path)
            else:
                # It's a file - add it to our list
                f["folder_path"] = path  # Add folder path info
                all_files.append(f)
                st.write(f"   ✅ Added file: {file_name} to results")
    
    # Start recursive scan from root
    scan_folder_recursively(folder_id)
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
    try:
        st.write("🔍 Checking for 01_Project_Plan and _metadata folders...")  # 👈 DEBUG LOG
        
        project_plan_id = get_gdrive_id_by_name("01_Project_Plan", PROJECT_ROOT_ID, is_folder=True)
        if not project_plan_id:
            st.warning("⚠️ Could not find 01_Project_Plan folder. Metadata features disabled.")
            return None
        
        st.write(f"✅ Found 01_Project_Plan folder: {project_plan_id}")  # 👈 DEBUG LOG
        
        metadata_folder_id = get_gdrive_id_by_name("_metadata", project_plan_id, is_folder=True)
        if not metadata_folder_id:
            st.warning("⚠️ Could not find 01_Project_Plan/_metadata folder. Metadata features disabled.")
            return None
        
        st.write(f"✅ Found _metadata folder: {metadata_folder_id}")  # 👈 DEBUG LOG
        
        return metadata_folder_id
    except Exception as e:
        st.warning(f"⚠️ Error accessing metadata folder: {e}")
        return None
