# PY Files/file_download.py
"""
Enterprise file download utilities for serving artifacts to users.
Provides secure download mechanisms for analysis results.
"""

import io
import os
from typing import Optional
import streamlit as st

def create_download_button(file_path: str, file_data: bytes, filename: str) -> str:
    """
    Create a Streamlit download button for file artifacts.
    
    Args:
        file_path: Dropbox path of the file
        file_data: File content as bytes
        filename: Display filename
        
    Returns:
        str: Download button HTML or message
    """
    try:
        if st and hasattr(st, 'download_button'):
            # Create download button in Streamlit
            st.download_button(
                label=f"📎 Download {filename}",
                data=file_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            return f"Download button created for {filename}"
        else:
            return f"File saved to cloud: {file_path}"
    except Exception as e:
        return f"File available at: {file_path} (Download: Contact admin)"


def get_file_download_info(artifact_path: str) -> dict:
    """
    Get download information for an artifact file.
    
    Args:
        artifact_path: Dropbox path to the artifact
        
    Returns:
        dict: Download information
    """
    filename = artifact_path.split("/")[-1] if "/" in artifact_path else artifact_path
    
    return {
        "filename": filename,
        "cloud_path": artifact_path,
        "access_instructions": [
            "Option 1: Open Dropbox app → Apps → Ethos LLM → navigate to file",
            "Option 2: Contact system administrator for direct access",
            "Option 3: Use Streamlit download button (if available)"
        ],
        "file_type": "Excel Workbook" if filename.endswith('.xlsx') else "CSV File"
    }


def format_download_message(artifact_path: str) -> str:
    """
    Format a user-friendly download message.
    
    Args:
        artifact_path: Path to the artifact
        
    Returns:
        str: Formatted message
    """
    info = get_file_download_info(artifact_path)
    
    message = f"""
**📊 Analysis Results Ready**

• **File**: {info['filename']} ({info['file_type']})
• **Location**: {info['cloud_path']}

**How to Access Your File:**
1. **Dropbox App**: Apps → Ethos LLM → navigate to file location  
2. **Web Browser**: Log into Dropbox → Apps → Ethos LLM folder
3. **Direct Request**: Contact your administrator for immediate access

*Note: File contains complete analysis results with all data points and calculations.*
"""
    
    return message.strip()
