# s3_utils.py
import os
import json
import io
from typing import List, Optional, Dict, Any
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Enterprise foundation imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Configuration from secrets.toml or environment
def _get_config():
    """Get S3 configuration from Streamlit secrets or environment variables."""
    if STREAMLIT_AVAILABLE and hasattr(st, "secrets"):
        return {
            "region": st.secrets.get("AWS_DEFAULT_REGION", "us-east-2"),
            "bucket": st.secrets.get("S3_BUCKET"),
            "prefix": st.secrets.get("S3_PREFIX", "").strip("/"),
            "access_key": st.secrets.get("AWS_ACCESS_KEY_ID"),
            "secret_key": st.secrets.get("AWS_SECRET_ACCESS_KEY")
        }
    else:
        return {
            "region": os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
            "bucket": os.getenv("S3_BUCKET"),
            "prefix": os.getenv("S3_PREFIX", "").strip("/"),
            "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY")
        }

config = _get_config()
AWS_REGION = config["region"]
S3_BUCKET = config["bucket"]
S3_PREFIX = config["prefix"]

_session = None
_client = None


def _get_session():
    global _session
    if _session is None:
        config = _get_config()
        _session = boto3.Session(
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"],
            region_name=config["region"],
        )
    return _session


def get_client():
    """Return a cached S3 client."""
    global _client
    if _client is None:
        _client = _get_session().client("s3")
    return _client


def _k(key: str) -> str:
    """Prefix key with S3_PREFIX if set."""
    key = key.lstrip("/")
    return f"{S3_PREFIX}/{key}" if S3_PREFIX else key


def ensure_prefix(prefix: str) -> str:
    """Normalize a folder-like prefix."""
    p = _k(prefix).strip("/")
    return f"{p}/" if p and not p.endswith("/") else p


def list_objects(prefix: str = "", max_keys: int = 1000) -> List[str]:
    """List object keys under a prefix."""
    s3 = get_client()
    pfx = ensure_prefix(prefix)
    keys = []
    kwargs = {"Bucket": S3_BUCKET, "Prefix": pfx, "MaxKeys": max_keys}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            keys.append(item["Key"])
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]
    return keys


def upload_json(obj: Dict[str, Any], key: str, indent: Optional[int] = None) -> str:
    """Upload a JSON object to S3."""
    s3 = get_client()
    body = json.dumps(obj, indent=indent).encode("utf-8")
    s3.put_object(Bucket=S3_BUCKET, Key=_k(key), Body=body, ContentType="application/json")
    return _k(key)


def download_json(key: str) -> Dict[str, Any]:
    """Download and parse JSON from S3."""
    s3 = get_client()
    resp = s3.get_object(Bucket=S3_BUCKET, Key=_k(key))
    return json.loads(resp["Body"].read().decode("utf-8"))


def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    """Upload arbitrary bytes to S3."""
    s3 = get_client()
    s3.put_object(Bucket=S3_BUCKET, Key=_k(key), Body=data, ContentType=content_type)
    return _k(key)


def download_bytes(key: str) -> bytes:
    """Download arbitrary bytes from S3."""
    s3 = get_client()
    resp = s3.get_object(Bucket=S3_BUCKET, Key=_k(key))
    return resp["Body"].read()


def append_jsonl_line(key: str, record: Dict[str, Any]) -> str:
    """
    Append a single JSONL record (newline-delimited JSON).
    Safe for low concurrency; for high-write rates use Kinesis/Firehose.
    """
    s3 = get_client()
    k = _k(key)
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        # Try to get existing object and append
        existing = s3.get_object(Bucket=S3_BUCKET, Key=k)["Body"].read()
        buf = io.BytesIO(existing + line)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            buf = io.BytesIO(line)
        else:
            raise
    s3.put_object(Bucket=S3_BUCKET, Key=k, Body=buf.getvalue(), ContentType="application/x-ndjson")
    return k


def exists(key: str) -> bool:
    s3 = get_client()
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=_k(key))
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def delete_object(key: str) -> None:
    s3 = get_client()
    s3.delete_object(Bucket=S3_BUCKET, Key=_k(key))


def upload_manifest(manifest_data: Dict[str, Any], manifest_key: str = None) -> str:
    """
    Upload a manifest file to S3 with enterprise metadata.
    
    Args:
        manifest_data: Manifest dictionary to upload
        manifest_key: Optional S3 key. If None, generates timestamped key.
        
    Returns:
        str: S3 key where manifest was uploaded
    """
    if manifest_key is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_key = f"manifests/sync_manifest_{timestamp}.json"
    
    # Enhance manifest with S3-specific metadata
    enhanced_manifest = manifest_data.copy()
    enhanced_manifest["cloud_sync"] = {
        "s3_bucket": S3_BUCKET,
        "s3_prefix": S3_PREFIX,
        "s3_region": AWS_REGION,
        "upload_timestamp": datetime.now().isoformat(),
        "manifest_key": _k(manifest_key)
    }
    
    return upload_json(enhanced_manifest, manifest_key, indent=2)


def list_manifests(prefix: str = "manifests/") -> List[Dict[str, Any]]:
    """
    List all manifest files in S3 with metadata.
    
    Args:
        prefix: S3 prefix to search for manifests
        
    Returns:
        List of manifest metadata dictionaries
    """
    s3 = get_client()
    manifests = []
    
    try:
        keys = list_objects(prefix)
        
        for key in keys:
            if key.endswith('.json') and 'manifest' in key.lower():
                try:
                    # Get object metadata
                    response = s3.head_object(Bucket=S3_BUCKET, Key=key)
                    
                    manifests.append({
                        "key": key,
                        "size": response.get("ContentLength", 0),
                        "last_modified": response.get("LastModified"),
                        "content_type": response.get("ContentType"),
                        "relative_key": key.replace(_k(""), "").lstrip("/")
                    })
                except ClientError:
                    # Skip files we can't access
                    continue
                    
    except Exception as e:
        print(f"Warning: Could not list manifests: {e}")
    
    # Sort by last modified, newest first
    manifests.sort(key=lambda x: x.get("last_modified", datetime.min), reverse=True)
    return manifests


def validate_s3_config() -> Dict[str, Any]:
    """
    Validate S3 configuration and connectivity.
    
    Returns:
        Dict with validation results
    """
    config = _get_config()
    validation = {
        "config_valid": False,
        "connectivity": False,
        "bucket_accessible": False,
        "errors": [],
        "config_source": "secrets.toml" if STREAMLIT_AVAILABLE else "environment"
    }
    
    # Check configuration completeness
    required_fields = ["bucket", "access_key", "secret_key", "region"]
    missing_fields = [field for field in required_fields if not config.get(field)]
    
    if missing_fields:
        validation["errors"].append(f"Missing configuration: {', '.join(missing_fields)}")
        return validation
    
    validation["config_valid"] = True
    
    # Test connectivity
    try:
        s3 = get_client()
        
        # Test basic connectivity
        s3.list_objects_v2(Bucket=config["bucket"], MaxKeys=1)
        validation["connectivity"] = True
        validation["bucket_accessible"] = True
        
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ["NoSuchBucket", "404"]:
            validation["errors"].append(f"Bucket '{config['bucket']}' does not exist")
        elif error_code in ["AccessDenied", "403"]:
            validation["errors"].append("Access denied - check credentials and permissions")
        else:
            validation["errors"].append(f"S3 error: {error_code}")
    except Exception as e:
        validation["errors"].append(f"Connection error: {str(e)}")
    
    return validation
