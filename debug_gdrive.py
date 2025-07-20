# --- debug_gdrive.py ---

import streamlit as st
from utils.gdrive import list_all_supported_files, PROJECT_ROOT_ID
from gdrive_utils import get_drive_service

st.title("🔍 Google Drive Debug Tool")

st.write("**Project Root ID:**", PROJECT_ROOT_ID)

try:
    # Test Google Drive service
    st.write("### Testing Google Drive Connection...")
    service = get_drive_service()
    st.success("✅ Google Drive service created successfully")
    
    # Test file listing
    st.write("### Testing File Listing...")
    files = list_all_supported_files()
    
    st.write(f"**Total files found:** {len(files)}")
    
    if files:
        st.write("### Files Found:")
        for i, file in enumerate(files[:10]):  # Show first 10 files
            st.write(f"{i+1}. **{file['name']}** ({file.get('mimeType', 'Unknown type')})")
        
        if len(files) > 10:
            st.write(f"... and {len(files) - 10} more files")
    else:
        st.warning("⚠️ No files found in Google Drive")
        
        # Test if we can access the root folder
        st.write("### Testing Root Folder Access...")
        try:
            folder_info = service.files().get(fileId=PROJECT_ROOT_ID).execute()
            st.write(f"Root folder name: {folder_info.get('name', 'Unknown')}")
            st.success("✅ Can access root folder")
        except Exception as e:
            st.error(f"❌ Cannot access root folder: {e}")

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.write("**Error details:**")
    st.code(str(e))
