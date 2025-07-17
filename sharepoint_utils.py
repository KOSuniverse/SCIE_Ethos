import json
import os
from datetime import datetime
from io import BytesIO
import streamlit as st
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.client_credential import ClientCredential

# --- AUTHENTICATE TO SHAREPOINT ---
def get_sharepoint_context():
    creds = st.secrets["sharepoint"]
    client_credentials = ClientCredential(
        creds["client_id"],
        creds["client_secret"]
    )
    return ClientContext(creds["site_url"]).with_credentials(client_credentials)

# --- FULL RELATIVE PATH UTILS ---
def build_full_metadata_path(filename):
    folder = st.secrets["sharepoint"]["metadata_folder"]
    return f"{folder}/{filename}.json"

def get_global_alias_path():
    return st.secrets["sharepoint"]["alias_file_path"]

def get_learned_answers_path():
    # Return the SharePoint path for learned_answers.json
    # Example:
    return st.secrets["sharepoint"]["learned_answers_path"]

# --- METADATA SIDELOAD ---
def load_metadata(filename):
    ctx = get_sharepoint_context()
    file_url = build_full_metadata_path(filename)
    file = ctx.web.get_file_by_server_relative_url(file_url).download()
    ctx.execute_query()
    return json.loads(file.content)

def save_metadata(filename, data):
    ctx = get_sharepoint_context()
    folder_path = st.secrets["sharepoint"]["metadata_folder"]
    json_str = json.dumps(data, indent=2).encode("utf-8")
    file_stream = BytesIO(json_str)
    ctx.web.get_folder_by_server_relative_url(folder_path).upload_file(f"{filename}.json", file_stream).execute_query()

# --- GLOBAL ALIAS DICT ---
def load_global_aliases():
    try:
        ctx = get_sharepoint_context()
        path = get_global_alias_path()
        buffer = BytesIO()
        ctx.web.get_file_by_server_relative_url(path).download(buffer).execute_query()
        buffer.seek(0)
        file_content = buffer.read().decode("utf-8")
        return json.loads(file_content)
    except Exception as e:
        st.error(f"SharePoint error: {e}")
        # Optionally, don't try to create the file if unauthorized
        if "401" in str(e):
            raise RuntimeError("SharePoint authentication failed. Check your credentials and permissions.")
        # If file doesn't exist, create a blank one
        try:
            ctx = get_sharepoint_context()
            path = os.path.dirname(get_global_alias_path())
            filename = os.path.basename(get_global_alias_path())
            empty = BytesIO(b"{}")
            ctx.web.get_folder_by_server_relative_url(path).upload_file(filename, empty).execute_query()
            return {}
        except Exception as e2:
            st.error(f"Failed to create blank alias file: {e2}")
            raise

def update_global_aliases(new_aliases):
    existing = load_global_aliases()
    updated = existing.copy()
    updated.update(new_aliases)

    # --- Save backup log with timestamp ---
    ctx = get_sharepoint_context()
    path = os.path.dirname(get_global_alias_path())
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_filename = f"alias_log_{timestamp}.json"
    log_stream = BytesIO(json.dumps(existing, indent=2).encode("utf-8"))
    ctx.web.get_folder_by_server_relative_url(path).upload_file(log_filename, log_stream).execute_query()

    # --- Save updated aliases ---
    filename = os.path.basename(get_global_alias_path())
    json_str = json.dumps(updated, indent=2).encode("utf-8")
    file_stream = BytesIO(json_str)
    ctx.web.get_folder_by_server_relative_url(path).upload_file(filename, file_stream).execute_query()

# --- LIST SUPPORTED DOCUMENTS ---
def list_all_supported_files(extensions=(".xlsx", ".pdf", ".docx", ".pptx")):
    ctx = get_sharepoint_context()
    root = ctx.web.get_folder_by_server_relative_url("Documents")
    file_list = []

    def traverse(folder):
        ctx.load(folder.files)
        ctx.load(folder.folders)
        ctx.execute_query()
        for file in folder.files:
            name = file.properties["Name"]
            if name.lower().endswith(extensions):
                file_list.append({
                    "name": name,
                    "relative_path": file.properties["ServerRelativeUrl"]
                })
        for subfolder in folder.folders:
            traverse(subfolder)

    traverse(root)
    return file_list

# --- DOWNLOAD A FILE FROM SHAREPOINT ---
def download_file(relative_path):
    ctx = get_sharepoint_context()
    file = ctx.web.get_file_by_server_relative_url(relative_path).download()
    ctx.execute_query()
    return BytesIO(file.content)

# --- LEARNED ANSWERS ---
def load_learned_answers():
    try:
        ctx = get_sharepoint_context()
        path = get_learned_answers_path()
        file = ctx.web.get_file_by_server_relative_url(path).download().execute_query()
        file_content = file.content
        return json.loads(file_content)
    except Exception:
        return {}

def save_learned_answers(data):
    ctx = get_sharepoint_context()
    path = os.path.dirname(get_learned_answers_path())
    filename = os.path.basename(get_learned_answers_path())
    json_str = json.dumps(data, indent=2).encode("utf-8")
    file_stream = BytesIO(json_str)
    ctx.web.get_folder_by_server_relative_url(path).upload_file(filename, file_stream).execute_query()
