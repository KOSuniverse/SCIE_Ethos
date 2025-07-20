# --- gdrive_utils.py ---

import json
import streamlit as st
from googleapiclient.discovery import build
from google.oauth2 import service_account
from io import BytesIO

# Supported file types
SUPPORTED_EXTENSIONS = {'.docx', '.xlsx', '.pptx', '.pdf', '.txt', '.csv'}

def get_drive_service():
    """Create and return Google Drive service using service account credentials from Streamlit secrets."""
    try:
        # Get credentials from Streamlit secrets
        credentials_dict = dict(st.secrets["gdrive_service_account"])
        
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        
        # Build and return the service
        service = build('drive', 'v3', credentials=credentials)
        return service
    
    except Exception as e:
        st.error(f"Failed to authenticate with Google Drive: {e}")
        st.stop()

def list_all_supported_files(folder_id=None, mime_type_filter=None):
    """
    List all supported files in Google Drive folder.
    
    Args:
        folder_id: Google Drive folder ID to search in (optional)
        mime_type_filter: Filter by specific MIME type (optional)
    
    Returns:
        List of file dictionaries with id, name, mimeType, etc.
    """
    service = get_drive_service()
    files = []
    page_token = None
    
    try:
        while True:
            # Build query
            query_parts = []
            
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")
            
            if mime_type_filter:
                query_parts.append(f"mimeType='{mime_type_filter}'")
            else:
                # Filter for supported file types
                supported_mimes = [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",       # .xlsx
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation", # .pptx
                    "application/pdf",                                                           # .pdf
                    "text/plain",                                                               # .txt
                    "text/csv",                                                                 # .csv
                    "application/vnd.google-apps.folder"                                        # folders
                ]
                mime_query = " or ".join([f"mimeType='{mime}'" for mime in supported_mimes])
                query_parts.append(f"({mime_query})")
            
            # Add trashed filter
            query_parts.append("trashed=false")
            
            query = " and ".join(query_parts)
            
            # Execute query
            results = service.files().list(
                q=query,
                pageSize=100,
                pageToken=page_token,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)"
            ).execute()
            
            items = results.get('files', [])
            files.extend(items)
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
                
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return []
    
    return files

def download_file(file_id):
    """
    Download a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        
    Returns:
        BytesIO object containing file content
    """
    service = get_drive_service()
    
    try:
        # Get file metadata first
        file_metadata = service.files().get(fileId=file_id).execute()
        
        # Download file content
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()
        
        return BytesIO(file_content)
        
    except Exception as e:
        st.error(f"Error downloading file {file_id}: {e}")
        return None

def get_file_last_modified(file_id):
    """
    Get the last modified time of a file.
    
    Args:
        file_id: Google Drive file ID
        
    Returns:
        ISO format timestamp string or None
    """
    service = get_drive_service()
    
    try:
        file_metadata = service.files().get(fileId=file_id, fields='modifiedTime').execute()
        return file_metadata.get('modifiedTime')
        
    except Exception as e:
        st.error(f"Error getting file modified time {file_id}: {e}")
        return None

def load_json_file(file_id):
    """
    Load and parse a JSON file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        
    Returns:
        Parsed JSON data or empty dict
    """
    try:
        file_stream = download_file(file_id)
        if file_stream:
            file_stream.seek(0)
            return json.load(file_stream)
        return {}
        
    except Exception as e:
        st.warning(f"Error loading JSON file {file_id}: {e}")
        return {}

def save_json_file(file_id, data):
    """
    Save JSON data to an existing file in Google Drive.
    
    Args:
        file_id: Google Drive file ID
        data: Data to save as JSON
    """
    service = get_drive_service()
    
    try:
        # Convert data to JSON bytes
        json_bytes = json.dumps(data, indent=2).encode('utf-8')
        media_body = BytesIO(json_bytes)
        
        # Update the file
        service.files().update(
            fileId=file_id,
            media_body=media_body
        ).execute()
        
    except Exception as e:
        st.error(f"Error saving JSON file {file_id}: {e}")

def create_json_file(folder_id, filename, data):
    """
    Create a new JSON file in Google Drive.
    
    Args:
        folder_id: Parent folder ID
        filename: Name for the new file
        data: Data to save as JSON
        
    Returns:
        File ID of created file or None
    """
    service = get_drive_service()
    
    try:
        # Convert data to JSON bytes
        json_bytes = json.dumps(data, indent=2).encode('utf-8')
        media_body = BytesIO(json_bytes)
        
        # File metadata
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        # Create the file
        file = service.files().create(
            body=file_metadata,
            media_body=media_body,
            fields='id'
        ).execute()
        
        return file.get('id')
        
    except Exception as e:
        st.error(f"Error creating JSON file {filename}: {e}")
        return None

def upload_file(folder_id, filename, file_content, mime_type):
    """
    Upload a file to Google Drive.
    
    Args:
        folder_id: Parent folder ID
        filename: Name for the file
        file_content: File content as bytes
        mime_type: MIME type of the file
        
    Returns:
        File ID of uploaded file or None
    """
    service = get_drive_service()
    
    try:
        media_body = BytesIO(file_content)
        
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        file = service.files().create(
            body=file_metadata,
            media_body=media_body,
            fields='id'
        ).execute()
        
        return file.get('id')
        
    except Exception as e:
        st.error(f"Error uploading file {filename}: {e}")
        return None

def get_folder_id_by_name(folder_name, parent_id=None):
    """
    Find a folder by name within a parent folder.
    
    Args:
        folder_name: Name of the folder to find
        parent_id: Parent folder ID (optional)
        
    Returns:
        Folder ID or None if not found
    """
    service = get_drive_service()
    
    try:
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        results = service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()
        
        items = results.get('files', [])
        if items:
            return items[0]['id']
        
        return None
        
    except Exception as e:
        st.error(f"Error finding folder {folder_name}: {e}")
        return None
