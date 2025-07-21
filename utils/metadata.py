import streamlit as st
import json
from gdrive_utils import (
    load_json_file, save_json_file, create_json_file,
    list_all_supported_files
)

def get_gdrive_id_by_name(name, parent_id, is_folder=False):
    files = list_all_supported_files(parent_id)
    for f in files:
        if f["name"] == name:
            if is_folder and f["mimeType"] == "application/vnd.google-apps.folder":
                return f["id"]
            if not is_folder and f["mimeType"] != "application/vnd.google-apps.folder":
                return f["id"]
    return None

def load_metadata(filename, metadata_folder_id):
    try:
        if not metadata_folder_id:
            return {}
        file_id = get_gdrive_id_by_name(f"{filename}.json", metadata_folder_id)
        return load_json_file(file_id) if file_id else {}
    except Exception:
        return {}

def save_metadata(filename, data, metadata_folder_id):
    try:
        if not metadata_folder_id:
            return
        json_filename = f"{filename}.json"
        file_id = get_gdrive_id_by_name(json_filename, metadata_folder_id)
        if file_id:
            save_json_file(file_id, data)
        else:
            create_json_file(metadata_folder_id, json_filename, data)
    except Exception as e:
        st.warning(f"Could not save metadata for {filename}: {e}")

def load_global_aliases(metadata_folder_id):
    try:
        if not metadata_folder_id:
            return {}
        file_id = get_gdrive_id_by_name("global_column_aliases.json", metadata_folder_id)
        return load_json_file(file_id) if file_id else {}
    except Exception:
        return {}

def update_global_aliases(new_aliases, metadata_folder_id):
    try:
        if not metadata_folder_id or not new_aliases:
            return
        file_id = get_gdrive_id_by_name("global_column_aliases.json", metadata_folder_id)
        if file_id:
            save_json_file(file_id, new_aliases)
        else:
            create_json_file(metadata_folder_id, "global_column_aliases.json", new_aliases)
    except Exception as e:
        st.warning(f"Could not update global column aliases: {e}")

def load_learned_answers(metadata_folder_id):
    try:
        if not metadata_folder_id:
            return {}
        file_id = get_gdrive_id_by_name("learned_answers.json", metadata_folder_id)
        return load_json_file(file_id) if file_id else {}
    except Exception:
        return {}

def save_learned_answers(data, metadata_folder_id):
    try:
        if not metadata_folder_id or not data:
            return
        file_id = get_gdrive_id_by_name("learned_answers.json", metadata_folder_id)
        if file_id:
            save_json_file(file_id, data)
        else:
            create_json_file(metadata_folder_id, "learned_answers.json", data)
    except Exception as e:
        st.warning(f"Could not save learned answers: {e}")

