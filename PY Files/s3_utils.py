# s3_utils.py
import os
import json
import io
from typing import List, Optional, Dict, Any
import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")

_session = None
_client = None


def _get_session():
    global _session
    if _session is None:
        _session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION,
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
