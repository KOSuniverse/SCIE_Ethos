# --- utils/metadata.py ---

import streamlit as st
import json
from utils.gdrive import get_gdrive_id_by_name, get_metadata_folder_id
from gdrive_utils import load_json_file, save_json_file

def load_metadata(filename):
    """Load metadata for a specific file."""
    metadata_folder_id = get_metadata_folder_id()
    file_id = get_gdrive_id_by_name(f"{filename}.json", metadata_folder_id)
    if not file_id:
        st.warning(f"Metadata file {filename}.json not found.")
        return {}
    return load_json_file(file_id)

def save_metadata(filename, data):
    """Save metadata for a specific file."""
    metadata_folder_id = get_metadata_folder_id()
    file_id = get_gdrive_id_by_name(f"{filename}.json", metadata_folder_id)
    if not file_id:
        st.warning(f"Metadata file {filename}.json not found. (Create logic not implemented.)")
        return
    save_json_file(file_id, data)

def load_global_aliases():
    """Load global column aliases."""
    metadata_folder_id = get_metadata_folder_id()
    file_id = get_gdrive_id_by_name("global_column_aliases.json", metadata_folder_id)
    if not file_id:
        st.warning("global_column_aliases.json not found in metadata folder.")
        return {}
    return load_json_file(file_id)

def update_global_aliases(new_aliases):
    """Update global column aliases."""
    metadata_folder_id = get_metadata_folder_id()
    file_id = get_gdrive_id_by_name("global_column_aliases.json", metadata_folder_id)
    if not file_id:
        st.warning("global_column_aliases.json not found in metadata folder.")
        return
    save_json_file(file_id, new_aliases)

def load_learned_answers():
    """Load previously learned answers."""
    metadata_folder_id = get_metadata_folder_id()
    file_id = get_gdrive_id_by_name("learned_answers.json", metadata_folder_id)
    if not file_id:
        return {}
    return load_json_file(file_id)

def save_learned_answers(data):
    """Save learned answers."""
    metadata_folder_id = get_metadata_folder_id()
    file_id = get_gdrive_id_by_name("learned_answers.json", metadata_folder_id)
    if not file_id:
        st.warning("learned_answers.json not found in metadata folder.")
        return
    save_json_file(file_id, data)
