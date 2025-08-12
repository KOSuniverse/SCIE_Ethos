# metadata_utils.py

import os
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

# Enterprise foundation imports
from constants import METADATA_DIR, PROJECT_ROOT
from path_utils import join_root, canon_path

# Cloud integration imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Assistant integration
try:
    from assistant_bridge import run_query as assistants_answer
    ASSISTANT_AVAILABLE = True
except ImportError:
    ASSISTANT_AVAILABLE = False
    def assistants_answer(query, context=""): return "Assistant not available"

def get_master_metadata_path() -> str:
    """
    Returns the standardized path to master metadata index using enterprise foundations.
    
    Returns:
        str: Full path to master_metadata_index.json in metadata folder
    """
    return join_root(METADATA_DIR, "master_metadata_index.json")

def load_master_metadata_index(metadata_path: Optional[str] = None) -> dict:
    """
    Loads the master metadata index from JSON file.
    Uses standardized metadata folder if no path specified.

    Args:
        metadata_path (str): Optional custom path. Uses standard path if None.

    Returns:
        dict: Parsed metadata dictionary.
    """
    if metadata_path is None:
        metadata_path = get_master_metadata_path()
    
    if not os.path.exists(metadata_path):
        return {"files": []}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_master_metadata_index(metadata_index: dict, metadata_path: Optional[str] = None):
    """
    Saves the updated metadata index to disk.
    Uses standardized metadata folder if no path specified.

    Args:
        metadata_index (dict): Metadata dictionary to write.
        metadata_path (str): Optional custom path. Uses standard path if None.
    """
    if metadata_path is None:
        metadata_path = get_master_metadata_path()
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_index, f, indent=2)

def find_file_metadata(metadata_index: dict, filename: str) -> dict:
    """
    Finds metadata for a specific file.

    Args:
        metadata_index (dict): Full metadata index.
        filename (str): Filename to locate.

    Returns:
        dict or None: File-level metadata or None.
    """
    for file_meta in metadata_index.get("files", []):
        if file_meta.get("filename") == filename:
            return file_meta
    return None


def _get_secrets_config() -> Dict[str, Any]:
    """Get configuration from Streamlit secrets or environment variables."""
    config = {}
    
    if STREAMLIT_AVAILABLE and hasattr(st, "secrets"):
        # Get Dropbox configuration
        config["dropbox"] = {
            "app_key": st.secrets.get("DROPBOX_APP_KEY"),
            "app_secret": st.secrets.get("DROPBOX_APP_SECRET"), 
            "refresh_token": st.secrets.get("DROPBOX_REFRESH_TOKEN"),
            "root": st.secrets.get("DROPBOX_ROOT", "")
        }
        
        # Get S3 configuration
        config["s3"] = {
            "bucket": st.secrets.get("S3_BUCKET"),
            "access_key": st.secrets.get("AWS_ACCESS_KEY_ID"),
            "secret_key": st.secrets.get("AWS_SECRET_ACCESS_KEY"),
            "region": st.secrets.get("AWS_DEFAULT_REGION", "us-east-2"),
            "prefix": st.secrets.get("S3_PREFIX", "")
        }
    else:
        # Fallback to environment variables
        config["dropbox"] = {
            "app_key": os.getenv("DROPBOX_APP_KEY"),
            "app_secret": os.getenv("DROPBOX_APP_SECRET"),
            "refresh_token": os.getenv("DROPBOX_REFRESH_TOKEN"),
            "root": os.getenv("DROPBOX_ROOT", "")
        }
        
        config["s3"] = {
            "bucket": os.getenv("S3_BUCKET"),
            "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "region": os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
            "prefix": os.getenv("S3_PREFIX", "")
        }
    
    return config


def generate_manifest_metadata(files_processed: List[Dict], operation_type: str = "sync") -> Dict[str, Any]:
    """
    Generate comprehensive manifest metadata for cloud sync operations.
    
    Args:
        files_processed: List of file metadata dictionaries
        operation_type: Type of operation (sync, upload, download, etc.)
        
    Returns:
        Dict containing manifest metadata with timestamps, file counts, and Assistant insights
    """
    timestamp = datetime.now().isoformat()
    
    manifest = {
        "operation": {
            "type": operation_type,
            "timestamp": timestamp,
            "source": "SCIE_Ethos_Enterprise",
            "version": "1.0"
        },
        "summary": {
            "total_files": len(files_processed),
            "file_types": {},
            "total_size_bytes": 0,
            "date_range": {"earliest": None, "latest": None}
        },
        "files": files_processed,
        "cloud_targets": {
            "dropbox_enabled": False,
            "s3_enabled": False
        }
    }
    
    # Analyze files for summary statistics
    for file_info in files_processed:
        # File type analysis
        file_type = file_info.get("file_type", "unknown")
        manifest["summary"]["file_types"][file_type] = manifest["summary"]["file_types"].get(file_type, 0) + 1
        
        # Size tracking
        size = file_info.get("size_bytes", 0)
        manifest["summary"]["total_size_bytes"] += size
        
        # Date tracking
        file_date = file_info.get("last_modified")
        if file_date:
            if not manifest["summary"]["date_range"]["earliest"] or file_date < manifest["summary"]["date_range"]["earliest"]:
                manifest["summary"]["date_range"]["earliest"] = file_date
            if not manifest["summary"]["date_range"]["latest"] or file_date > manifest["summary"]["date_range"]["latest"]:
                manifest["summary"]["date_range"]["latest"] = file_date
    
    # Check cloud configuration
    config = _get_secrets_config()
    manifest["cloud_targets"]["dropbox_enabled"] = bool(config["dropbox"]["app_key"] and config["dropbox"]["refresh_token"])
    manifest["cloud_targets"]["s3_enabled"] = bool(config["s3"]["bucket"] and config["s3"]["access_key"])
    
    # Assistant-enhanced insights
    if ASSISTANT_AVAILABLE and files_processed:
        try:
            context = {
                "operation_type": operation_type,
                "file_count": len(files_processed),
                "file_types": manifest["summary"]["file_types"],
                "total_size_mb": manifest["summary"]["total_size_bytes"] / (1024 * 1024),
                "cloud_targets": manifest["cloud_targets"]
            }
            
            insights = assistants_answer(
                f"Analyze this {operation_type} operation with {len(files_processed)} files. "
                f"Provide insights about data quality, sync patterns, and potential issues. "
                f"Focus on supply chain data management best practices.",
                context=json.dumps(context, default=str)[:1500]
            )
            
            manifest["ai_insights"] = {
                "analysis": insights,
                "generated_at": timestamp,
                "assistant_available": True
            }
            
        except Exception as e:
            manifest["ai_insights"] = {
                "analysis": f"Assistant analysis failed: {str(e)}",
                "generated_at": timestamp,
                "assistant_available": False
            }
    else:
        manifest["ai_insights"] = {
            "analysis": "Assistant not available for this operation",
            "generated_at": timestamp,
            "assistant_available": False
        }
    
    return manifest


def sync_manifest_to_cloud(manifest: Dict[str, Any], manifest_name: str = "sync_manifest") -> Dict[str, str]:
    """
    Sync generated manifest to both Dropbox and S3 cloud storage.
    
    Args:
        manifest: Manifest metadata dictionary
        manifest_name: Base name for the manifest file
        
    Returns:
        Dict with sync results for each cloud provider
    """
    results = {
        "dropbox": {"success": False, "path": None, "error": None},
        "s3": {"success": False, "path": None, "error": None}
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_filename = f"{manifest_name}_{timestamp}.json"
    
    # Sync to Dropbox
    try:
        from dbx_utils import upload_json as dbx_upload_json
        
        config = _get_secrets_config()
        if config["dropbox"]["app_key"] and config["dropbox"]["refresh_token"]:
            dropbox_path = f"/manifests/{manifest_filename}"
            dbx_upload_json(dropbox_path, manifest)
            
            results["dropbox"]["success"] = True
            results["dropbox"]["path"] = dropbox_path
            
    except Exception as e:
        results["dropbox"]["error"] = str(e)
    
    # Sync to S3
    try:
        from s3_utils import upload_json as s3_upload_json
        
        config = _get_secrets_config()
        if config["s3"]["bucket"] and config["s3"]["access_key"]:
            s3_key = f"manifests/{manifest_filename}"
            s3_upload_json(manifest, s3_key, indent=2)
            
            results["s3"]["success"] = True
            results["s3"]["path"] = s3_key
            
    except Exception as e:
        results["s3"]["error"] = str(e)
    
    return results


def enhance_file_metadata(file_path: str, base_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhance file metadata with Assistant-driven analysis and enterprise context.
    
    Args:
        file_path: Path to the file to analyze
        base_metadata: Optional base metadata to enhance
        
    Returns:
        Enhanced metadata dictionary
    """
    if base_metadata is None:
        base_metadata = {}
    
    enhanced = base_metadata.copy()
    enhanced.update({
        "enhancement_timestamp": datetime.now().isoformat(),
        "enterprise_analysis": {
            "path_canonical": canon_path(file_path),
            "path_safe": True,
            "project_relative": os.path.relpath(file_path, PROJECT_ROOT) if PROJECT_ROOT in file_path else None
        }
    })
    
    # File system analysis
    if os.path.exists(file_path):
        stat = os.stat(file_path)
        enhanced["file_system"] = {
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
        }
        
        # Assistant-driven content analysis
        if ASSISTANT_AVAILABLE:
            try:
                file_context = {
                    "filename": os.path.basename(file_path),
                    "extension": os.path.splitext(file_path)[1],
                    "size_mb": enhanced["file_system"]["size_mb"],
                    "path": enhanced["enterprise_analysis"]["project_relative"]
                }
                
                analysis = assistants_answer(
                    f"Analyze this supply chain data file: {os.path.basename(file_path)}. "
                    f"Based on the filename and context, what type of supply chain data does this likely contain? "
                    f"Provide data classification and handling recommendations.",
                    context=json.dumps(file_context, default=str)[:1000]
                )
                
                enhanced["ai_analysis"] = {
                    "content_classification": analysis,
                    "confidence": "high" if len(analysis) > 100 else "medium",
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                enhanced["ai_analysis"] = {
                    "content_classification": f"Analysis failed: {str(e)}",
                    "confidence": "none",
                    "analysis_timestamp": datetime.now().isoformat()
                }
    
    return enhanced
