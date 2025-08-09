import os, json, hashlib, tempfile, mimetypes
from datetime import datetime
from typing import List, Tuple

# --- SDK version guard (no client instantiation) ---
import openai as _openai
try:
    from packaging import version as _v
except Exception:
    _v = None  # fallback if packaging not available

MIN_VER = "1.52.0"
_current = getattr(_openai, "__version__", "0")

def _too_old(vstr: str, minv: str) -> bool:
    if _v:
        try:
            return _v.parse(vstr) < _v.parse(minv)
        except Exception:
            pass
    # fallback: naive compare on dotted version strings
    def _parts(s): return [int(p) for p in s.split(".") if p.isdigit()]
    return _parts(vstr) < _parts(minv)

if _too_old(_current, MIN_VER):
    raise RuntimeError(
        f"OpenAI SDK too old ({_current}). Please set openai>={MIN_VER} in requirements.txt and redeploy."
    )

# Secrets loader (Streamlit or env/.env)
try:
    import streamlit as st
    SECRETS = st.secrets
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    SECRETS = os.environ

import dropbox
from openai import OpenAI


# Resolve repo root regardless of CWD
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(HERE, ".."))

# Config
ALLOWED_EXTS = {".pdf", ".docx", ".xlsx", ".csv", ".txt"}
MANIFEST_PATH = os.path.join(REPO_DIR, "config", "dropbox_manifest.json")
VECTOR_STORE_META_PATH = os.path.join(REPO_DIR, "config", "vector_store.json")
ASSISTANT_META_PATH = os.path.join(REPO_DIR, "config", "assistant.json")  # optional fallback

def must(key: str) -> str:
    v = SECRETS.get(key)
    if not v:
        raise EnvironmentError(f"Missing required secret: {key}")
    return v

def get_optional(key: str, default: str = "") -> str:
    v = SECRETS.get(key)
    return v if v is not None else default

# Required secrets
OPENAI_API_KEY = must("OPENAI_API_KEY")
ASSISTANT_ID = SECRETS.get("ASSISTANT_ID")  # may be None; will fallback to assistant.json
DBX_APP_KEY = must("DROPBOX_APP_KEY")
DBX_APP_SECRET = must("DROPBOX_APP_SECRET")
DBX_REFRESH_TOKEN = must("DROPBOX_REFRESH_TOKEN")

# Optional
DROPBOX_ROOT = get_optional("DROPBOX_ROOT", "")  # e.g. "/Project_Root" or "" for root
OPENAI_MODEL = get_optional("OPENAI_MODEL", "gpt-4o")

# Init OpenAI client (reads OPENAI_API_KEY from env OR set explicitly)
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
client = OpenAI()

def resolve_assistant_id() -> str:
    if ASSISTANT_ID:
        return ASSISTANT_ID
    if os.path.exists(ASSISTANT_META_PATH):
        with open(ASSISTANT_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["assistant_id"]
    raise FileNotFoundError(
        "ASSISTANT_ID is not set in secrets and config/assistant.json was not found."
    )

def init_dropbox() -> dropbox.Dropbox:
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=DBX_REFRESH_TOKEN,
        app_key=DBX_APP_KEY,
        app_secret=DBX_APP_SECRET,
    )
    # fail fast if bad auth/path
    dbx.files_list_folder(DROPBOX_ROOT or "", recursive=False)
    return dbx

def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def list_dropbox_files(dbx: dropbox.Dropbox, path: str):
    files = []
    result = dbx.files_list_folder(path or "", recursive=True)
    files.extend(result.entries)
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        files.extend(result.entries)
    return [f for f in files if isinstance(f, dropbox.files.FileMetadata)]

# --- Vector store (new OpenAI SDK: beta namespace) ---
def get_or_create_vector_store() -> str:
    meta = load_json(VECTOR_STORE_META_PATH, {})
    vs_id = meta.get("vector_store_id")
    if vs_id:
        try:
            client.beta.vector_stores.retrieve(vs_id)
            return vs_id
        except Exception:
            pass  # recreate if deleted
    vs = client.beta.vector_stores.create(name="SCIE Ethos Source Docs")
    save_json(VECTOR_STORE_META_PATH, {"vector_store_id": vs.id})
    return vs.id

def attach_vector_store_to_assistant(assistant_id: str, vs_id: str):
    # Ensure assistant exists and attach file_search tool with the vector store
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

def _upload_batch_to_vector_store(vs_id: str, file_tuples: List[Tuple[str, bytes]]):
    """
    file_tuples: list of (filename, content_bytes)
    """
    # SDK accepts list of file-like objects or (name, BytesIO, mime) tuples.
    import io as _io
    files = []
    for name, data in file_tuples:
        mime, _ = mimetypes.guess_type(name)
        files.append((name, _io.BytesIO(data), mime or "application/octet-stream"))

    batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vs_id,
        files=files,
    )
    return batch.status  # usually "completed"

def sync_dropbox_to_assistant(batch_size: int = 10):
    assistant_id = resolve_assistant_id()
    dbx = init_dropbox()
    manifest = load_json(MANIFEST_PATH, {})
    dropbox_files = list_dropbox_files(dbx, DROPBOX_ROOT)

    vs_id = get_or_create_vector_store()
    attach_vector_store_to_assistant(assistant_id, vs_id)

    uploaded, skipped = 0, 0
    buffer: List[Tuple[str, bytes, str, str]] = []  # (name, content, key, hash)

    for fmeta in dropbox_files:
        ext = os.path.splitext(fmeta.name)[1].lower()
        if ext not in ALLOWED_EXTS:
            continue

        _, resp = dbx.files_download(fmeta.path_lower)
        content = resp.content
        h = file_hash(content)
        key = fmeta.path_lower  # stable key

        if key in manifest and manifest[key]["hash"] == h:
            skipped += 1
            continue

        buffer.append((fmeta.name, content, key, h))

        if len(buffer) >= batch_size:
            _upload_batch_to_vector_store(vs_id, [(n, c) for (n, c, _, _) in buffer])
            # ✅ per-file manifest updates (was bugged before)
            now = datetime.utcnow().isoformat()
            for _, _, k, hh in buffer:
                manifest[k] = {"hash": hh, "last_sync": now}
                uploaded += 1
            buffer = []

    if buffer:
        _upload_batch_to_vector_store(vs_id, [(n, c) for (n, c, _, _) in buffer])
        now = datetime.utcnow().isoformat()
        for _, _, k, hh in buffer:
            manifest[k] = {"hash": hh, "last_sync": now}
            uploaded += 1

    save_json(MANIFEST_PATH, manifest)
    print(f"✅ Sync complete. Uploaded: {uploaded}, Skipped: {skipped}")
    print(f"Vector store: {vs_id} (saved in {VECTOR_STORE_META_PATH})")

if __name__ == "__main__":
    sync_dropbox_to_assistant()




