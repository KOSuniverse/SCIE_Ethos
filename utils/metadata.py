# --- utils/metadata.py ---

import streamlit as st
import json
from utils.gdrive import get_gdrive_id_by_name, get_metadata_folder_id
from gdrive_utils import load_json_file, save_json_file, create_json_file

def load_metadata(filename):
    """Load metadata for a specific file."""
    try:
        metadata_folder_id = get_metadata_folder_id()
        if not metadata_folder_id:
            return {}  # No metadata folder available
        
        file_id = get_gdrive_id_by_name(f"{filename}.json", metadata_folder_id)
        if not file_id:
            # Metadata doesn't exist yet - that's normal for new files
            return {}
        return load_json_file(file_id)
    except Exception:
        return {}

def save_metadata(filename, data):
    """Save metadata for a specific file - create JSON file if it doesn't exist."""
    try:
        st.write(f"📝 Saving metadata for {filename}")  # 🔍 DEBUG LOG
        st.write(f"📤 Inside save_metadata() for {filename}")  # 🔍 DEBUG LOG
        metadata_folder_id = get_metadata_folder_id()
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

def load_global_aliases():
    """Load global column aliases."""
    try:
        metadata_folder_id = get_metadata_folder_id()
        if not metadata_folder_id:
            return {}  # No metadata folder available
        
        file_id = get_gdrive_id_by_name("global_column_aliases.json", metadata_folder_id)
        if not file_id:
            # File doesn't exist yet - that's OK, return empty dict
            return {}
        return load_json_file(file_id)
    except Exception:
        return {}

def update_global_aliases(new_aliases):
    """Update global column aliases - create file if it doesn't exist."""
    try:
        if not new_aliases:  # Don't save empty aliases
            return
        metadata_folder_id = get_metadata_folder_id()
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

def load_learned_answers():
    """Load previously learned answers."""
    try:
        metadata_folder_id = get_metadata_folder_id()
        if not metadata_folder_id:
            return {}  # No metadata folder available
        
        file_id = get_gdrive_id_by_name("learned_answers.json", metadata_folder_id)
        if not file_id:
            return {}
        return load_json_file(file_id)
    except Exception:
        return {}

def save_learned_answers(data):
    """Save learned answers - create file if it doesn't exist."""
    try:
        if not data:  # Don't save empty data
            return
        metadata_folder_id = get_metadata_folder_id()
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
