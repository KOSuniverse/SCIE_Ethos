# sidecars.py
# sidecars.py (Dropbox-first, XLSX-only, flat folders; no env gating)
import os, re, json, io, hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
try:
    from constants import DATA_ROOT, PROJECT_ROOT
except Exception:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", ".")
    DATA_ROOT = "/04_Data"

RAW_DIR            = f"{DATA_ROOT}/00_Raw_Files"
CLEANSED_DIR       = f"{DATA_ROOT}/01_Cleansed_Files"
EDA_CHARTS_DIR     = f"{DATA_ROOT}/02_EDA_Charts"
SUMMARIES_DIR      = f"{DATA_ROOT}/03_Summaries"   # <- summaries (flat)
METADATA_DIR       = f"{DATA_ROOT}/04_Metadata"    # <- metadata + EDA + masters (flat)
PER_SHEET_DATA_ROOT= f"{CLEANSED_DIR}/split"

MASTER_META_PATH   = f"{METADATA_DIR}/master_metadata.jsonl"
MASTER_SUM_PATH    = f"{METADATA_DIR}/master_summaries.jsonl"
MASTER_EDA_PATH    = f"{METADATA_DIR}/master_eda.jsonl"

# -------------------------------------------------------------------------
# Dropbox / S3 adapters (Dropbox-first, import unconditionally)
# -------------------------------------------------------------------------
_dbx = None
_dbx_client = None
try:
    import dbx_utils as _dbx  # must expose: upload_bytes, upload_json, read_file_bytes, _get_dbx_client
    try:
        _dbx_client = _dbx._get_dbx_client()
    except Exception:
        _dbx_client = None
except Exception:
    _dbx = None
    _dbx_client = None

_s3 = None
try:
    import s3_utils as _s3   # optional; if present will mirror writes
except Exception:
    _s3 = None

def backend_info() -> str:
    if _dbx is not None:
        return "dropbox"
    return "local-fallback"

# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------
def _slug(s: str) -> str:
    s = str(s or "").strip().replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120] or "sheet"

def _key(filename: str, sheet: str) -> str:
    base = _slug(os.path.splitext(os.path.basename(filename))[0])
    sh   = _slug(sheet)
    return f"{base}__{sh}"

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _hash_df(df) -> str:
    try:
        head = df.head(1000).astype(str)
        blob = "|".join(map(str, df.columns)) + "\n" + head.to_csv(index=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    except Exception:
        return ""

def _put_bytes(path: str, data: bytes):
    if _dbx:
        _dbx.upload_bytes(path, data)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    if _s3:
        try: _s3.upload_bytes(data, path.strip("/"))
        except Exception: pass

def _put_json(path: str, obj: Dict[str, Any]):
    payload = json.dumps(obj, indent=2, ensure_ascii=False)
    if _dbx and hasattr(_dbx, "upload_json"):
        _dbx.upload_json(path, obj)
    else:
        _put_bytes(path, payload.encode("utf-8"))

def _put_text(path: str, text: str):
    _put_bytes(path, text.encode("utf-8"))

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if _dbx:
        try:
            data = _dbx.read_file_bytes(path)
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return None

def _list_json(dir_path: str) -> List[str]:
    # Dropbox SDK listing via dbx_client
    if _dbx_client:
        try:
            import dropbox
            dbx = _dbx_client
            entries: List[dropbox.files.Metadata] = []
            resp = dbx.files_list_folder(dir_path)
            entries.extend(resp.entries)
            while resp.has_more:
                resp = dbx.files_list_folder_continue(resp.cursor)
                entries.extend(resp.entries)
            return [e.path_lower for e in entries
                    if isinstance(e, dropbox.files.FileMetadata) and e.name.lower().endswith(".json")]
        except Exception:
            return []
    # Local fallback
    if not os.path.isdir(dir_path): return []
    return [os.path.join(dir_path, n) for n in os.listdir(dir_path) if n.lower().endswith(".json")]

# -------------------------------------------------------------------------
# Writers (Dropbox-first)
# -------------------------------------------------------------------------
def write_metadata(doc: Dict[str, Any], df=None) -> str:
    """
    Per-sheet metadata JSON into 04_Data/04_Metadata/.
    <file>__<sheet>__<stage>.json with _kind="metadata"
    """
    file_stem = doc.get("source_file") or "file"
    sheet     = doc.get("sheet_name")  or "sheet"
    stage     = doc.get("stage")       or "raw"
    key = _key(file_stem, sheet)
    path = f"{METADATA_DIR}/{key}__{stage}.json"

    payload = dict(doc)
    payload.setdefault("_kind", "metadata")
    payload.setdefault("_created_utc", _now())

    if df is not None:
        cols_idx = getattr(df, "columns", None)
        cols_list = list(map(str, list(cols_idx))) if cols_idx is not None else []
        nrows = int(getattr(df, "shape", [0])[0] or 0)
        payload.setdefault("row_count", nrows)
        payload.setdefault("columns", cols_list)
        payload.setdefault("hash", _hash_df(df))

    _put_json(path, payload)
    return path

def write_summary(summary: Dict[str, Any] | str, source_file: str, sheet: str, stage: str, kind: str = "summary") -> str:
    """
    Per-sheet summary into 04_Data/03_Summaries/.
    If str -> .txt/.md + index JSON; if dict -> JSON with _kind="summary".
    """
    key = _key(source_file, sheet)
    if isinstance(summary, str):
        ext = "md" if summary.strip().startswith("#") else "txt"
        txt_path = f"{SUMMARIES_DIR}/{key}__{stage}.{ext}"
        _put_text(txt_path, summary)
        _put_json(f"{SUMMARIES_DIR}/{key}__{stage}.json", {
            "_kind": kind, "stage": stage, "path": txt_path,
            "source_file": source_file, "sheet_name": sheet,
            "_created_utc": _now()
        })
        return txt_path
    else:
        jpath = f"{SUMMARIES_DIR}/{key}__{stage}.json"
        doc = dict(summary)
        doc.setdefault("_kind", kind)
        doc.setdefault("stage", stage)
        doc.setdefault("source_file", source_file)
        doc.setdefault("sheet_name", sheet)
        doc.setdefault("_created_utc", _now())
        _put_json(jpath, doc)
        return jpath

def write_eda(eda_obj: Dict[str, Any], source_file: str, sheet: str, stage: str = "eda") -> str:
    """
    EDA JSON into 04_Data/04_Metadata/ with _kind="eda".
    <file>__<sheet>__eda.json
    """
    key = _key(source_file, sheet)
    path = f"{METADATA_DIR}/{key}__{stage}.json"
    doc = dict(eda_obj or {})
    doc.setdefault("_kind", "eda")
    doc.setdefault("stage", stage)
    doc.setdefault("source_file", source_file)
    doc.setdefault("sheet_name", sheet)
    doc.setdefault("_created_utc", _now())
    _put_json(path, doc)
    return path

def save_cleansed_table(df, source_file: str, sheet: str, normalized_type: str = "unclassified") -> str:
    """
    XLSX-only cleansed write to /04_Data/01_Cleansed_Files/split/<type>/<file>__<sheet>.xlsx
    """
    sub = _slug(normalized_type or "unclassified")
    key = _key(source_file, sheet)
    path = f"{PER_SHEET_DATA_ROOT}/{sub}/{key}.xlsx"

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        sname = (sheet or "Sheet")[:31]
        df.to_excel(writer, index=False, sheet_name=sname)
    _put_bytes(path, bio.getvalue())
    return path

# -------------------------------------------------------------------------
# Master rollups (JSONL) in 04_Data/04_Metadata/
# -------------------------------------------------------------------------
def _rollup_jsonl_kind(dir_path: str, out_path: str, keep_kind: str) -> str:
    items = _list_json(dir_path)
    lines = []
    for p in sorted(items):
        obj = _read_json(p)
        if obj and obj.get("_kind") == keep_kind:
            lines.append(json.dumps(obj, ensure_ascii=False))
    _put_text(out_path, "\n".join(lines))
    return out_path

def rebuild_master_indexes() -> Dict[str, str]:
    meta = _rollup_jsonl_kind(METADATA_DIR,  MASTER_META_PATH, keep_kind="metadata")
    summ = _rollup_jsonl_kind(SUMMARIES_DIR, MASTER_SUM_PATH,  keep_kind="summary")
    eda  = _rollup_jsonl_kind(METADATA_DIR,  MASTER_EDA_PATH,  keep_kind="eda")
    return {"metadata": meta, "summaries": summ, "eda": eda}

# Export for metadata_utils back-compat
MASTER_META_PATH_EXPORT = MASTER_META_PATH
