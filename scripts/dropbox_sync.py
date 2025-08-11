# scripts/dropbox_sync.py
import os, json, hashlib, mimetypes
from datetime import datetime
from typing import List, Tuple

# --- OpenAI SDK guard ---
import openai as _openai
try:
    from packaging import version as _v
except Exception:
    _v = None
MIN_VER = "1.52.0"
def _too_old(vstr: str, minv: str) -> bool:
    if _v:
        try: return _v.parse(vstr) < _v.parse(minv)
        except Exception: pass
    def _p(s): return [int(p) for p in s.split(".") if p.replace(".", "").isdigit()]
    return _p(vstr) < _p(minv)
if _too_old(getattr(_openai, "__version__", "0"), MIN_VER):
    raise RuntimeError(f"OpenAI SDK too old ({getattr(_openai,'__version__','0')}). Please set openai>={MIN_VER}.")

# --- Secrets loader (Streamlit or env/.env) ---
try:
    import streamlit as st
    SECRETS = st.secrets
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    SECRETS = os.environ

import dropbox
from openai import OpenAI

# ----- Config / constants -----
# Extensions allowed in Dropbox sync (all data files we support)
ALL_SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".xlsx", ".xls"}

# Extensions supported by OpenAI File Search (no CSV/Excel - we handle those separately)
OPENAI_FILE_SEARCH_EXTS = {".pdf", ".docx", ".pptx", ".txt", ".md"}

# Extensions we support for tabular data (handled by our tools, not OpenAI File Search)
TABULAR_EXTS = {".csv", ".xlsx", ".xls"}

# Folders to exclude from OpenAI File Search sync (these contain generated artifacts)
EXCLUDE_FOLDERS = {
    "04_data/02_eda_charts",      # Generated charts
    "04_data/03_summaries",       # Generated analysis artifacts (CSV files)
    "04_data/05_merged_comparisons",  # Generated comparisons
    "logs",                       # Log files
    "__pycache__",               # Python cache
}

def should_exclude_file(file_path: str, file_name: str, ext: str) -> tuple[bool, str]:
    """
    Determine if a file should be excluded from OpenAI File Search sync.
    Returns (should_exclude, reason)
    
    Architecture principle: "tabular files use Assistant File Search"
    REALITY: OpenAI File Search doesn't support CSV/Excel (platform limitation)
    SOLUTION: Tabular files available via our tools, documents via OpenAI File Search
    """
    # Normalize path for comparison
    path_lower = file_path.lower().replace("\\", "/")
    
    # Exclude by folder location (generated artifacts)
    for exclude_folder in EXCLUDE_FOLDERS:
        if exclude_folder in path_lower:
            return True, f"Generated artifact folder: {exclude_folder}"
    
    # Exclude tabular files (OpenAI limitation) - our tools handle these
    if ext in TABULAR_EXTS:
        return True, "Tabular file (OpenAI File Search doesn't support CSV/Excel)"
    
    # Exclude unsupported extensions 
    if ext not in OPENAI_FILE_SEARCH_EXTS:
        return True, f"Unsupported extension: {ext}"
    
    return False, "File approved for OpenAI File Search"

def must(key: str) -> str:
    v = SECRETS.get(key)
    if not v: raise EnvironmentError(f"Missing required secret: {key}")
    return v
def get_optional(key: str, default: str = "") -> str:
    v = SECRETS.get(key); return v if v is not None else default

# Required
OPENAI_API_KEY = must("OPENAI_API_KEY")
DBX_APP_KEY     = must("DROPBOX_APP_KEY")
DBX_APP_SECRET  = must("DROPBOX_APP_SECRET")
DBX_REFRESH_TOKEN = must("DROPBOX_REFRESH_TOKEN")
ASSISTANT_ID = SECRETS.get("ASSISTANT_ID")  # optional; can fall back to assistant.json if you keep one in Dropbox later

# Optional
DROPBOX_ROOT = get_optional("DROPBOX_ROOT", "").strip("/")   # e.g. "Project_Root"
OPENAI_MODEL = get_optional("OPENAI_MODEL", "gpt-4o")

# Dropbox config folder (APP-ROOT RELATIVE; no "/Apps/Ethos LLM" prefix)
DBX_CONFIG_DIR = f"/{DROPBOX_ROOT}/config" if DROPBOX_ROOT else "/config"
DBX_MANIFEST_PATH = f"{DBX_CONFIG_DIR}/dropbox_manifest.json"
DBX_VECTOR_META_PATH = f"{DBX_CONFIG_DIR}/vector_store.json"

# Init OpenAI client
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
client = OpenAI()

# ---------------- Vector Store adapter ----------------
class _VS:
    def __init__(self, client: OpenAI):
        self.client = client
        cand = [
            getattr(getattr(client, "beta", object()), "vector_stores", None),
            getattr(getattr(client, "beta", object()), "vectorstores", None),
            getattr(client, "vector_stores", None),
            getattr(client, "vectorstores", None),
        ]
        self.ns = next((c for c in cand if c is not None), None)
        if self.ns is None:
            raise AttributeError("OpenAI vector store API not found on this SDK.")
        self.file_batches = getattr(self.ns, "file_batches", None) or getattr(self.ns, "fileBatches", None)
        if self.file_batches is None:
            raise AttributeError("Vector store 'file_batches' API not found on this SDK.")
    def create(self, **kwargs): return self.ns.create(**kwargs)
    def retrieve(self, vector_store_id: str): return self.ns.retrieve(vector_store_id)
    def upload_batch_and_poll(self, vector_store_id: str, files):
        if hasattr(self.file_batches, "upload_and_poll"):
            return self.file_batches.upload_and_poll(vector_store_id=vector_store_id, files=files)
        if hasattr(self.file_batches, "upload"):
            batch = self.file_batches.upload(vector_store_id=vector_store_id, files=files)
            return getattr(batch, "status", "completed")
        raise AttributeError("No compatible upload method found on vector store file_batches.")

# ---------------- Dropbox helpers ----------------
def init_dropbox() -> dropbox.Dropbox:
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=DBX_REFRESH_TOKEN,
        app_key=DBX_APP_KEY,
        app_secret=DBX_APP_SECRET,
    )
    dbx.files_list_folder(DROPBOX_ROOT and f"/{DROPBOX_ROOT}" or "", recursive=False)  # fail fast
    return dbx

def _ensure_dbx_folder(dbx: dropbox.Dropbox, folder: str):
    folder = (folder or "/").rstrip("/") or "/"
    if folder == "/": return
    try:
        dbx.files_get_metadata(folder)
    except dropbox.exceptions.ApiError:
        dbx.files_create_folder_v2(folder, autorename=False)

def dbx_read_json(path: str):
    try:
        dbx = init_dropbox()
        _, res = dbx.files_download(path)
        return json.loads(res.content.decode("utf-8"))
    except Exception:
        return None

def dbx_write_json(path: str, data: dict):
    dbx = init_dropbox()
    _ensure_dbx_folder(dbx, os.path.dirname(path.rstrip("/")) or "/")
    dbx.files_upload(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
                     path, mode=dropbox.files.WriteMode.overwrite, mute=True)

def list_dropbox_files(dbx: dropbox.Dropbox, path: str):
    start = path and f"/{path.strip('/')}" or ""
    files = []
    res = dbx.files_list_folder(start, recursive=True)
    files.extend(res.entries)
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        files.extend(res.entries)
    return [f for f in files if isinstance(f, dropbox.files.FileMetadata)]

# ---------------- Assistant / Vector store ----------------
def resolve_assistant_id() -> str:
    if ASSISTANT_ID:
        return ASSISTANT_ID
    raise FileNotFoundError("ASSISTANT_ID not set in secrets.")

def get_or_create_vector_store() -> str:
    vs_meta = dbx_read_json(DBX_VECTOR_META_PATH) or {}
    vs_id = vs_meta.get("vector_store_id")
    vs = _VS(client)
    if vs_id:
        try:
            vs.retrieve(vs_id); return vs_id
        except Exception:
            pass  # recreate if missing
    created = vs.create(name="SCIE Ethos Source Docs")
    new_id = getattr(created, "id", None) or (isinstance(created, dict) and created.get("id"))
    if not new_id:
        raise RuntimeError("Could not obtain vector_store.id from create().")
    dbx_write_json(DBX_VECTOR_META_PATH, {"vector_store_id": new_id})
    return new_id

def attach_vector_store_to_assistant(assistant_id: str, vs_id: str):
    a = client.beta.assistants.retrieve(assistant_id)
    existing_ids = []
    try:
        tr = a.tool_resources or {}
        existing_ids = (tr.get("file_search") or {}).get("vector_store_ids", []) or []
    except Exception:
        existing_ids = []
    new_ids = list(dict.fromkeys(existing_ids + [vs_id]))
    client.beta.assistants.update(
        assistant_id=assistant_id,
        model=OPENAI_MODEL,
        tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
        tool_resources={"file_search": {"vector_store_ids": new_ids}},
    )

def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def _upload_batch_to_vector_store(vs_id: str, file_tuples: List[Tuple[str, bytes]]):
    import io as _io
    files = []
    for name, data in file_tuples:
        mime, _ = mimetypes.guess_type(name)
        files.append((name, _io.BytesIO(data), mime or "application/octet-stream"))
    vs = _VS(client)
    return vs.upload_batch_and_poll(vector_store_id=vs_id, files=files)

# ---------------- Main sync ----------------
def sync_dropbox_to_assistant(batch_size: int = 10):
    assistant_id = resolve_assistant_id()
    dbx = init_dropbox()
    manifest = dbx_read_json(DBX_MANIFEST_PATH) or {}
    dropbox_files = list_dropbox_files(dbx, DROPBOX_ROOT)

    vs_id = get_or_create_vector_store()
    attach_vector_store_to_assistant(assistant_id, vs_id)

    uploaded = skipped = excluded_artifacts = excluded_tabular = unsupported = 0
    buffer: List[Tuple[str, bytes, str, str]] = []  # (name, content, key, hash)

    for fmeta in dropbox_files:
        ext = os.path.splitext(fmeta.name)[1].lower()
        
        # Check if file should be excluded
        should_exclude, reason = should_exclude_file(fmeta.path_lower, fmeta.name, ext)
        
        if should_exclude:
            if "artifact" in reason.lower():
                excluded_artifacts += 1
                print(f"📁 Skipping artifact: {fmeta.name} ({reason})")
            elif "tabular" in reason.lower():
                excluded_tabular += 1
                print(f"📊 Skipping tabular file: {fmeta.name} ({reason})")
            else:
                unsupported += 1
                print(f"❌ Skipping unsupported: {fmeta.name} ({reason})")
            continue
            
        # File approved for OpenAI File Search
        _, resp = dbx.files_download(fmeta.path_lower)
        content = resp.content
        h = file_hash(content)
        key = fmeta.path_lower
        if key in manifest and manifest[key]["hash"] == h:
            skipped += 1; continue
        buffer.append((fmeta.name, content, key, h))
        if len(buffer) >= batch_size:
            _upload_batch_to_vector_store(vs_id, [(n, c) for (n, c, _, _) in buffer])
            now = datetime.utcnow().isoformat()
            for _, _, k, hh in buffer:
                manifest[k] = {"hash": hh, "last_sync": now}; uploaded += 1
            buffer = []

    if buffer:
        _upload_batch_to_vector_store(vs_id, [(n, c) for (n, c, _, _) in buffer])
        now = datetime.utcnow().isoformat()
        for _, _, k, hh in buffer:
            manifest[k] = {"hash": hh, "last_sync": now}; uploaded += 1

    dbx_write_json(DBX_MANIFEST_PATH, manifest)
    print(f"✅ Sync complete!")
    print(f"   📤 Uploaded to OpenAI: {uploaded}")
    print(f"   ⏭️  Skipped (unchanged): {skipped}")
    print(f"   📁 Excluded (artifacts): {excluded_artifacts}")
    print(f"   📊 Excluded (tabular data): {excluded_tabular}")
    print(f"   ❌ Excluded (unsupported): {unsupported}")
    print(f"   🔗 Vector store: {vs_id}")
    print(f"   💾 Manifest saved: {DBX_VECTOR_META_PATH}")

if __name__ == "__main__":
    sync_dropbox_to_assistant()






