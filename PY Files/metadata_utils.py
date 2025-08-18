# metadata_utils.py

# metadata_utils.py
# Full drop-in: per-sheet sidecars (metadata/summaries/EDA) + master rollups,
# while keeping your existing manifest/cloud/enhance utilities and
# backward-compatible master index helpers.

import os
import json
from typing import Optional, Dict, List, Any
from datetime import datetime

# ---- Project constants / path helpers (back-compat) ----
try:
    from constants import METADATA_DIR, PROJECT_ROOT
except Exception:
    METADATA_DIR = os.path.join(os.getenv("PROJECT_ROOT", "."), "04_Data", "04_Metadata")
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", ".")

try:
    from path_utils import join_root, canon_path
except Exception:
    def join_root(*parts: str) -> str:
        return os.path.join(*parts)
    def canon_path(p: str) -> str:
        return os.path.abspath(p)

# ---- Streamlit optional ----
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None  # type: ignore

# ---- Assistant optional ----
try:
    from assistant_bridge import run_query as assistants_answer
    ASSISTANT_AVAILABLE = True
except ImportError:
    ASSISTANT_AVAILABLE = False
    def assistants_answer(query, context: str = "") -> str:  # type: ignore
        return "Assistant not available"

# ---- Sidecars: per-sheet writers + rollups ----
from sidecars import (
    write_metadata as _sc_write_metadata,
    write_summary as _sc_write_summary,
    write_eda as _sc_write_eda,
    save_cleansed_table as _sc_save_cleansed_table,
    rebuild_master_indexes as _sc_rebuild_master_indexes,
    MASTER_META_PATH as _SC_MASTER_META_PATH,
)

# =====================================================================================
# Backward-compatible "master metadata index" helpers
# =====================================================================================

def get_master_metadata_path() -> str:
    """
    Returns the canonical path to the master metadata index.
    For back-compat this now points at the JSONL rollup produced by sidecars.
    """
    try:
        return _SC_MASTER_META_PATH
    except Exception:
        return join_root(METADATA_DIR, "master_metadata_index.json")

def load_master_metadata_index(metadata_path: Optional[str] = None) -> dict:
    """
    Loads the master metadata index.

    - If pointed to sidecars JSONL, returns {"files": [...]} by reading each JSON line.
    - If pointed to legacy JSON, returns it as-is.
    """
    path = metadata_path or get_master_metadata_path()
    if not os.path.exists(path):
        return {"files": []}

    if path.endswith(".jsonl"):
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
        return {"files": items}

    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
            if isinstance(obj, dict) and "files" in obj:
                return obj
            return {"files": obj if isinstance(obj, list) else []}
        except Exception:
            return {"files": []}

def save_master_metadata_index(metadata_index: dict, metadata_path: Optional[str] = None):
    """
    Saves the updated metadata index.

    Preferred: call rollup_all_master_indexes() to rebuild JSONL from sidecars.
    """
    path = metadata_path or get_master_metadata_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith(".jsonl"):
        items = metadata_index.get("files", []) if isinstance(metadata_index, dict) else []
        with open(path, "w", encoding="utf-8") as w:
            for obj in items:
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata_index, f, indent=2, ensure_ascii=False)

def find_file_metadata(metadata_index: dict, filename: str) -> Optional[dict]:
    """
    Finds metadata for a specific file in a loaded master index.
    """
    for file_meta in metadata_index.get("files", []):
        name = file_meta.get("source_file") or file_meta.get("filename")
        if name and os.path.basename(name) == os.path.basename(filename):
            return file_meta
    return None

# =====================================================================================
# New per-sheet, per-stage helpers (thin wrappers over sidecars)
# =====================================================================================

def save_per_sheet_metadata(filename: str, sheet: str, metadata: Dict[str, Any], df=None) -> str:
    doc = dict(metadata)
    doc.setdefault("source_file", filename)
    doc.setdefault("sheet_name", sheet)
    doc.setdefault("stage", metadata.get("stage", "raw"))
    return _sc_write_metadata(doc, df=df)

def save_per_sheet_summary(filename: str, sheet: str, summary: Dict[str, Any] | str,
                           stage: str = "raw", kind: str = "summary") -> str:
    return _sc_write_summary(summary, filename, sheet, stage, kind=kind)

def save_per_sheet_eda(filename: str, sheet: str, eda_obj: Dict[str, Any]) -> str:
    return _sc_write_eda(eda_obj, filename, sheet, stage="eda")

def save_cleansed_table(df, filename: str, sheet: str,
                        normalized_type: str = "unclassified") -> str:
    # XLSX only
    return _sc_save_cleansed_table(df, filename, sheet, normalized_type)

def rollup_all_master_indexes() -> Dict[str, str]:
    return _sc_rebuild_master_indexes()

# =====================================================================================
# Cloud/manifest utilities (preserved)
# =====================================================================================

def _get_secrets_config() -> Dict[str, Any]:
    """Get configuration from Streamlit secrets or environment variables."""
    config: Dict[str, Any] = {}
    if STREAMLIT_AVAILABLE and hasattr(st, "secrets"):
        config["dropbox"] = {
            "app_key": st.secrets.get("DROPBOX_APP_KEY"),
            "app_secret": st.secrets.get("DROPBOX_APP_SECRET"),
            "refresh_token": st.secrets.get("DROPBOX_REFRESH_TOKEN"),
            "root": st.secrets.get("DROPBOX_ROOT", "")
        }
        config["s3"] = {
            "bucket": st.secrets.get("S3_BUCKET"),
            "access_key": st.secrets.get("AWS_ACCESS_KEY_ID"),
            "secret_key": st.secrets.get("AWS_SECRET_ACCESS_KEY"),
            "region": st.secrets.get("AWS_DEFAULT_REGION", "us-east-2"),
            "prefix": st.secrets.get("S3_PREFIX", "")
        }
    else:
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

def generate_manifest_metadata(files_processed: List[Dict[str, Any]],
                               operation_type: str = "sync") -> Dict[str, Any]:
    timestamp = datetime.now().isoformat()
    manifest: Dict[str, Any] = {
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
        "cloud_targets": {"dropbox_enabled": False, "s3_enabled": False}
    }

    for file_info in files_processed:
        ftype = file_info.get("file_type", "unknown")
        manifest["summary"]["file_types"][ftype] = manifest["summary"]["file_types"].get(ftype, 0) + 1
        size = file_info.get("size_bytes", 0)
        manifest["summary"]["total_size_bytes"] += size
        fdate = file_info.get("last_modified")
        if fdate:
            if not manifest["summary"]["date_range"]["earliest"] or fdate < manifest["summary"]["date_range"]["earliest"]:
                manifest["summary"]["date_range"]["earliest"] = fdate
            if not manifest["summary"]["date_range"]["latest"] or fdate > manifest["summary"]["date_range"]["latest"]:
                manifest["summary"]["date_range"]["latest"] = fdate

    cfg = _get_secrets_config()
    manifest["cloud_targets"]["dropbox_enabled"] = bool(cfg["dropbox"]["app_key"] and cfg["dropbox"]["refresh_token"])
    manifest["cloud_targets"]["s3_enabled"] = bool(cfg["s3"]["bucket"] and cfg["s3"]["access_key"])

    if ASSISTANT_AVAILABLE and files_processed:
        try:
            context = {
                "operation_type": operation_type,
                "file_count": len(files_processed),
                "file_types": manifest["summary"]["file_types"],
                "total_size_mb": manifest["summary"]["total_size_bytes"] / (1024 * 1024),
                "cloud_targets": manifest["cloud_targets"],
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
    results = {"dropbox": {"success": False, "path": None, "error": None},
               "s3": {"success": False, "path": None, "error": None}}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_filename = f"{manifest_name}_{timestamp}.json"

    # Dropbox
    try:
        from dbx_utils import upload_json as dbx_upload_json
        cfg = _get_secrets_config()
        if cfg["dropbox"]["app_key"] and cfg["dropbox"]["refresh_token"]:
            dropbox_path = f"/manifests/{manifest_filename}"
            dbx_upload_json(dropbox_path, manifest)
            results["dropbox"]["success"] = True
            results["dropbox"]["path"] = dropbox_path
    except Exception as e:
        results["dropbox"]["error"] = str(e)

    # S3
    try:
        from s3_utils import upload_json as s3_upload_json
        cfg = _get_secrets_config()
        if cfg["s3"]["bucket"] and cfg["s3"]["access_key"]:
            s3_key = f"manifests/{manifest_filename}"
            s3_upload_json(manifest, s3_key, indent=2)
            results["s3"]["success"] = True
            results["s3"]["path"] = s3_key
    except Exception as e:
        results["s3"]["error"] = str(e)

    return results

def enhance_file_metadata(file_path: str, base_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_metadata = base_metadata or {}
    enhanced: Dict[str, Any] = {**base_metadata}
    enhanced.update({
        "enhancement_timestamp": datetime.now().isoformat(),
        "enterprise_analysis": {
            "path_canonical": canon_path(file_path),
            "path_safe": True,
            "project_relative": os.path.relpath(file_path, PROJECT_ROOT) if PROJECT_ROOT and PROJECT_ROOT in file_path else None
        }
    })

    if os.path.exists(file_path):
        stat = os.stat(file_path)
        enhanced["file_system"] = {
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
        }

        if ASSISTANT_AVAILABLE:
            try:
                context = {
                    "filename": os.path.basename(file_path),
                    "extension": os.path.splitext(file_path)[1],
                    "size_mb": enhanced["file_system"]["size_mb"],
                    "path": enhanced["enterprise_analysis"]["project_relative"]
                }
                analysis = assistants_answer(
                    f"Analyze this supply chain data file: {os.path.basename(file_path)}. "
                    f"Based on the filename and context, what type of supply chain data does this likely contain? "
                    f"Provide data classification and handling recommendations.",
                    context=json.dumps(context, default=str)[:1000]
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
