# --- utils/metadata.py ---

import streamlit as st
import json
# Import from the correct paths for your project structure
from gdrive_utils import (
    load_json_file, save_json_file, create_json_file, 
    list_all_supported_files
)

def get_gdrive_id_by_name(name, parent_id, is_folder=False):
    """Find a file or folder by name within a parent folder."""
    files = list_all_supported_files(parent_id)
    for f in files:
        if f["name"] == name:
            if is_folder and f["mimeType"] == "application/vnd.google-apps.folder":
                return f["id"]
            if not is_folder and f["mimeType"] != "application/vnd.google-apps.folder":
                return f["id"]
    return None

def load_metadata(filename, metadata_folder_id):
    """Load metadata for a specific file."""
    try:
        if not metadata_folder_id:
            return {}  # No metadata folder available
        
        file_id = get_gdrive_id_by_name(f"{filename}.json", metadata_folder_id)
        if not file_id:
            # Metadata doesn't exist yet - that's normal for new files
            return {}
        return load_json_file(file_id)
    except Exception:
        return {}

def save_metadata(filename, data, metadata_folder_id):
    """Save metadata for a specific file - create JSON file if it doesn't exist."""
    try:
        st.write(f"📝 Saving metadata for {filename}")  # 🔍 DEBUG LOG
        st.write(f"📤 Inside save_metadata() for {filename}")  # 🔍 DEBUG LOG
        if not metadata_folder_id:
            st.write(f"❌ No metadata folder available for {filename}")  # 🔍 DEBUG LOG
            return  # No metadata folder available, skip saving
        
        json_filename = f"{filename}.json"
        file_id = get_gdrive_id_by_name(json_filename, metadata_folder_id)
        
        if file_id:
            # File exists - update it
            save_json_file(file_id, data)
            st.success(f"✅ Updated metadata for {filename}")
        else:
            # File doesn't exist - create it
            new_file_id = create_json_file(metadata_folder_id, json_filename, data)
            if new_file_id:
                st.success(f"🆕 Created metadata for {filename}")
            else:
                st.warning(f"⚠️ Could not create metadata file for {filename}")
    except Exception as e:
        st.warning(f"Could not save metadata for {filename}: {e}")

def load_global_aliases(metadata_folder_id):
    """Load global column aliases."""
    try:
        if not metadata_folder_id:
            return {}  # No metadata folder available
        
        file_id = get_gdrive_id_by_name("global_column_aliases.json", metadata_folder_id)
        if not file_id:
            # File doesn't exist yet - that's OK, return empty dict
            return {}
        return load_json_file(file_id)
    except Exception:
        return {}

def update_global_aliases(new_aliases, metadata_folder_id):
    """Update global column aliases - create file if it doesn't exist."""
    try:
        if not new_aliases:  # Don't save empty aliases
            return
        if not metadata_folder_id:
            return  # No metadata folder available, skip saving
        
        file_id = get_gdrive_id_by_name("global_column_aliases.json", metadata_folder_id)
        
        if file_id:
            # File exists - update it
            save_json_file(file_id, new_aliases)
            st.success("✅ Updated global column aliases")
        else:
            # File doesn't exist - create it
            new_file_id = create_json_file(metadata_folder_id, "global_column_aliases.json", new_aliases)
            if new_file_id:
                st.success("🆕 Created global column aliases file")
    except Exception as e:
        st.warning(f"Could not save aliases: {e}")

def load_learned_answers(metadata_folder_id):
    """Load previously learned answers."""
    try:
        if not metadata_folder_id:
            return {}  # No metadata folder available
        
        file_id = get_gdrive_id_by_name("learned_answers.json", metadata_folder_id)
        if not file_id:
            return {}
        return load_json_file(file_id)
    except Exception:
        return {}

def save_learned_answers(data, metadata_folder_id):
    """Save learned answers - create file if it doesn't exist."""
    try:
        if not data:  # Don't save empty data
            return
        if not metadata_folder_id:
            return  # No metadata folder available, skip saving
        
        file_id = get_gdrive_id_by_name("learned_answers.json", metadata_folder_id)
        
        if file_id:
            # File exists - update it
            save_json_file(file_id, data)
            st.success("✅ Updated learned answers")
        else:
            # File doesn't exist - create it
            new_file_id = create_json_file(metadata_folder_id, "learned_answers.json", data)
            if new_file_id:
                st.success("🆕 Created learned answers file")
    except Exception as e:
        st.warning(f"Could not save learned answers: {e}")
