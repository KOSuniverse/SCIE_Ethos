# PY Files/dropbox_kb_sync.py
"""
Dropbox Knowledge Base and Data File Sync Helper

Provides file listing and caching for Chat Assistant integration.
Lists documents from KB and light data files without heavy processing.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

# Cache settings
CACHE_FILE = "dropbox_kb_cache.json"
CACHE_TTL_MINUTES = 45  # 45 minute TTL

# File type filters
KB_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.md', '.txt', '.eml', '.msg', '.rtf'}
DATA_EXTENSIONS = {'.xlsx', '.csv'}

# System files to include (these are valuable for KB operations)
SYSTEM_FILES = {
    'document_index.faiss',
    'docstore.pkl', 
    'manifest.json'
}

# Files to exclude (OS and temp files)
EXCLUDE_FILES = {
    '.DS_Store',
    'Thumbs.db',
    'desktop.ini',
    '.gitignore'
}

# Folders to exclude from document scanning (but not from system file detection)
EXCLUDE_FOLDERS = {
    '__pycache__',
    '.git',
    '.svn',
    'node_modules'
}

def _is_cache_valid(cache_path: str) -> bool:
    """Check if cache file exists and is within TTL."""
    if not os.path.exists(cache_path):
        return False
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        cache_time = cache_data.get('timestamp', 0)
        current_time = time.time()
        ttl_seconds = CACHE_TTL_MINUTES * 60
        
        return (current_time - cache_time) < ttl_seconds
    except Exception:
        return False

def _save_cache(cache_path: str, data: Dict[str, Any]) -> None:
    """Save data to cache with timestamp."""
    try:
        cache_data = {
            'timestamp': time.time(),
            'data': data
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

def _load_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    """Load data from cache if valid."""
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data.get('data')
    except Exception:
        return None

def _should_include_file(file_path: str, allowed_extensions: set, include_system_files: bool = False) -> bool:
    """Check if file should be included based on extension and exclusions."""
    file_name = os.path.basename(file_path)
    file_ext = Path(file_path).suffix.lower()
    
    # Always exclude OS/temp files
    if file_name in EXCLUDE_FILES:
        return False
    
    # Include system files if requested (FAISS index, docstore, etc.)
    if include_system_files and file_name in SYSTEM_FILES:
        return True
    
    # Check extension for regular documents
    return file_ext in allowed_extensions

def _should_include_folder(folder_name: str) -> bool:
    """Check if folder should be traversed."""
    return folder_name not in EXCLUDE_FOLDERS and not folder_name.startswith('.')

def _list_files_recursive(root_path: str, allowed_extensions: set, max_files: int = 50, include_system_files: bool = False) -> List[Dict[str, Any]]:
    """
    Recursively list files with metadata, sorted by modification time (newest first).
    
    Returns list of dicts with: path, name, size, modified_time
    """
    try:
        # Import Dropbox utilities
        from dbx_utils import list_data_files, get_file_info
        
        files = []
        
        # Use existing Dropbox utilities if available
        try:
            # Get file list from Dropbox
            all_files = list_data_files(root_path, recursive=True)
            
            for file_info in all_files:
                file_path = file_info.get('path_lower', file_info.get('path', ''))
                
                if _should_include_file(file_path, allowed_extensions, include_system_files):
                    # Get additional file metadata
                    try:
                        size = file_info.get('size', 0)
                        modified = file_info.get('server_modified', file_info.get('modified', ''))
                        
                        # Extract folder structure info
                        relative_path = file_path.replace(root_path, '').lstrip('/')
                        folder_path = os.path.dirname(relative_path) if os.path.dirname(relative_path) else 'root'
                        
                        files.append({
                            'path': file_path,
                            'name': os.path.basename(file_path),
                            'folder': folder_path,
                            'size': size,
                            'modified_time': modified,
                            'type': Path(file_path).suffix.lower(),
                            'is_system_file': os.path.basename(file_path) in SYSTEM_FILES
                        })
                    except Exception as e:
                        print(f"Warning: Could not get metadata for {file_path}: {e}")
                        # Add basic info even if metadata fails
                        relative_path = file_path.replace(root_path, '').lstrip('/')
                        folder_path = os.path.dirname(relative_path) if os.path.dirname(relative_path) else 'root'
                        
                        files.append({
                            'path': file_path,
                            'name': os.path.basename(file_path),
                            'folder': folder_path,
                            'size': 0,
                            'modified_time': '',
                            'type': Path(file_path).suffix.lower(),
                            'is_system_file': os.path.basename(file_path) in SYSTEM_FILES
                        })
                
                # Limit results for performance
                if len(files) >= max_files:
                    break
            
            # Sort by modification time (newest first) if available
            files.sort(key=lambda x: x.get('modified_time', ''), reverse=True)
            
        except Exception as e:
            print(f"Warning: Dropbox file listing failed: {e}")
            # Fallback to empty list rather than crash
            files = []
        
        return files[:max_files]
        
    except ImportError:
        print("Warning: dbx_utils not available, returning empty file list")
        return []
    except Exception as e:
        print(f"Error listing files from {root_path}: {e}")
        return []

def list_kb_candidates(kb_root: str, data_root: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    List knowledge base documents and light data files from Dropbox.
    
    Args:
        kb_root: Path to KB root (e.g., "/Project_Root/06_LLM_Knowledge_Base")
        data_root: Path to data root (e.g., "/Project_Root/04_Data")
    
    Returns:
        Dict with 'kb_docs' and 'data_files' lists containing file metadata
    """
    cache_path = CACHE_FILE
    
    # Check cache first
    if _is_cache_valid(cache_path):
        cached_data = _load_cache(cache_path)
        if cached_data:
            print("Using cached KB candidates")
            return cached_data
    
    print("Refreshing KB candidates from Dropbox...")
    
    # List KB documents (including system files like FAISS index)
    kb_docs = _list_files_recursive(kb_root, KB_EXTENSIONS, max_files=30, include_system_files=True)
    
    # List data files from cleansed folder
    cleansed_path = f"{data_root.rstrip('/')}/01_Cleansed_Files"
    data_files = _list_files_recursive(cleansed_path, DATA_EXTENSIONS, max_files=12)
    
    result = {
        'kb_docs': kb_docs,
        'data_files': data_files
    }
    
    # Cache the results
    _save_cache(cache_path, result)
    
    print(f"Found {len(kb_docs)} KB documents and {len(data_files)} data files")
    
    return result

def get_file_hash(file_path: str) -> str:
    """
    Generate a hash for a file path for deduplication.
    Uses path + size as hash input for efficiency.
    """
    try:
        # For now, use path as hash - could be enhanced with actual file content hash
        return hashlib.md5(file_path.encode('utf-8')).hexdigest()
    except Exception:
        return file_path.replace('/', '_').replace('\\', '_')

def get_kb_system_files(kb_docs: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract system files (FAISS index, docstore) from KB documents list.
    
    Returns dict mapping system file types to their paths.
    """
    system_files = {}
    
    for doc in kb_docs:
        if doc.get('is_system_file'):
            file_name = doc['name']
            if file_name == 'document_index.faiss':
                system_files['faiss_index'] = doc['path']
            elif file_name == 'docstore.pkl':
                system_files['docstore'] = doc['path']
            elif file_name == 'manifest.json':
                system_files['manifest'] = doc['path']
    
    return system_files

def get_folder_structure(kb_docs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize documents by folder structure.
    
    Returns dict mapping folder names to lists of documents.
    """
    folders = {}
    
    for doc in kb_docs:
        if not doc.get('is_system_file'):  # Skip system files
            folder = doc.get('folder', 'root')
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(doc)
    
    return folders

def clear_cache() -> None:
    """Clear the KB candidates cache."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print("KB candidates cache cleared")
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")

# For testing/debugging
if __name__ == "__main__":
    # Test with environment variables
    kb_path = os.getenv("KB_DBX_PATH", "/Project_Root/06_LLM_Knowledge_Base")
    data_path = os.getenv("DATA_DBX_PATH", "/Project_Root/04_Data")
    
    print(f"Testing with KB: {kb_path}, Data: {data_path}")
    
    candidates = list_kb_candidates(kb_path, data_path)
    
    print(f"\nKB Documents ({len(candidates['kb_docs'])}):")
    for doc in candidates['kb_docs'][:5]:  # Show first 5
        print(f"  - {doc['name']} ({doc['size']} bytes)")
    
    print(f"\nData Files ({len(candidates['data_files'])}):")
    for file in candidates['data_files'][:5]:  # Show first 5
        print(f"  - {file['name']} ({file['size']} bytes)")
