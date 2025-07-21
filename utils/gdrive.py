import streamlit as st
from gdrive_utils import (
    list_all_supported_files as _list_all_supported_files,
    download_file as _download_file,
    get_file_last_modified as _get_file_last_modified,
    load_json_file, save_json_file, get_drive_service
)

PROJECT_ROOT_ID = "1rhT8zhCu4FGy1YUxXABXhUYV92iZwlAE"  # Root Shared Drive folder

def get_gdrive_id_by_name(name, parent_id, is_folder=False):
    files = _list_all_supported_files(parent_id)
    for f in files:
        if f["name"] == name:
            if is_folder and f["mimeType"] == "application/vnd.google-apps.folder":
                return f["id"]
            if not is_folder and f["mimeType"] != "application/vnd.google-apps.folder":
                return f["id"]
    return None

def list_all_supported_files(folder_id=None):
    if folder_id is None:
        folder_id = PROJECT_ROOT_ID
    all_files = []

    def scan_folder_recursively(current_folder_id, path=""):
        files_in_folder = _list_all_supported_files(current_folder_id)
        for f in files_in_folder:
            file_name = f["name"]
            mime_type = f.get("mimeType", "")
            if file_name.startswith("Raw_Data"):
                continue
            if mime_type == "application/vnd.google-apps.folder":
                scan_folder_recursively(f["id"], f"{path}/{file_name}")
            else:
                f["folder_path"] = path
                all_files.append(f)

    scan_folder_recursively(folder_id)
    return all_files

def download_file(file_id):
    return _download_file(file_id)

def get_file_last_modified(file_id):
    return _get_file_last_modified(file_id)

def get_models_folder_id():
    data_folder_id = get_gdrive_id_by_name("04_Data", PROJECT_ROOT_ID, is_folder=True)
    if not data_folder_id:
        st.error("❌ Could not find 04_Data folder.")
        st.stop()
    models_folder_id = get_gdrive_id_by_name("Models", data_folder_id, is_folder=True)
    if not models_folder_id:
        st.error("❌ Could not find Models subfolder.")
        st.stop()
    return models_folder_id

def get_metadata_folder_id():
    try:
        project_plan_id = get_gdrive_id_by_name("01_Project_Plan", PROJECT_ROOT_ID, is_folder=True)
        if not project_plan_id:
            st.warning("⚠️ Could not find 01_Project_Plan folder.")
            return None
        metadata_folder_id = get_gdrive_id_by_name("_metadata", project_plan_id, is_folder=True)
        if not metadata_folder_id:
            st.warning("⚠️ Could not find _metadata folder.")
            return None
        return metadata_folder_id
    except Exception as e:
        st.warning(f"⚠️ Error accessing metadata folder: {e}")
        return None

