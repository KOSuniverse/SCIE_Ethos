import json
import os
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
    ctx = get_sharepoint_context()
    file_url = get_global_alias_path()
    try:
        file = ctx.web.get_file_by_server_relative_url(file_url).download()
        ctx.execute_query()
        return json.loads(file.content)
    except Exception as e:
        print(f"⚠️ global_column_aliases.json not found, returning empty dict. Error: {e}")
        return {}

def update_global_aliases(new_aliases):
    existing = load_global_aliases()
    existing.update(new_aliases)
    ctx = get_sharepoint_context()
    path = os.path.dirname(get_global_alias_path())
    filename = os.path.basename(get_global_alias_path())
    json_str = json.dumps(existing, indent=2).encode("utf-8")
    file_stream = BytesIO(json_str)
    ctx.web.get_folder_by_server_relative_url(path).upload_file(filename, file_stream).execute_query()

# --- LIST SUPPORTED DOCUMENTS ---
def list_all_supported_files(extensions=(".xlsx", ".pdf", ".docx", ".pptx")):
    ctx = get_sharepoint_context()
    root = ctx.web.get_folder_by_server_relative_url("Shared Documents")
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
