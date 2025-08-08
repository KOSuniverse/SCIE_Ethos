# scripts/dropbox_sync.py
import os, json, hashlib, tempfile
from datetime import datetime
from typing import List, Tuple

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

# --- Vector store (new OpenAI SDK) ---
def get_or_create_vector_store() -> str:
    meta = load_json(VECTOR_STORE_META_PATH, {})
    vs_id = meta.get("vector_store_id")
    if vs_id:
        try:
            client.vector_stores.retrieve(vs_id)
            return vs_id
        except Exception:
            pass  # recreate if deleted
    vs = client.vector_stores.create(name="SCIE Ethos Source Docs")
    save_json(VECTOR_STORE_META_PATH, {"vector_store_id": vs.id})
    return vs.id

def attach_vector_store_to_assistant(assistant_id: str, vs_id: str):
    a = client.assistants.retrieve(assistant_id)
    tr = a.tool_resources or {}
    existing_ids = []
    try:
        existing_ids = tr.get("file_search", {}).get("vector_store_ids", [])
    except Exception:
        pass
    if vs_id not in (existing_ids or []):
        new_ids = (existing_ids or []) + [vs_id]
        client.assistants.update(
            assistant_id=assistant_id,
            tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
            tool_resources={"file_search": {"vector_store_ids": new_ids}},
        )

def batch_upload_files_to_vector_store(vs_id: str, file_tuples: List[Tuple[str, bytes]]):
    """file_tuples: list of (filename, content_bytes)."""
    temp_paths, handles = [], []
    try:
        for name, data in file_tuples:
            fd, tmp = tempfile.mkstemp()
            with os.fdopen(fd, "wb") as w:
                w.write(data)
            base, ext = os.path.splitext(name)
            final_p = tmp + ext
            os.rename(tmp, final_p)
            temp_paths.append(final_p)

        for p in temp_paths:
            handles.append(open(p, "rb"))

        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs_id,
            files=handles,
        )
        # batch.status should be "completed" here
    finally:
        for h in handles:
            try: h.close()
            except: pass
        for p in temp_paths:
            try: os.remove(p)
            except: pass

def sync_dropbox_to_assistant():
    assistant_id = resolve_assistant_id()
    dbx = init_dropbox()
    manifest = load_json(MANIFEST_PATH, {})
    dropbox_files = list_dropbox_files(dbx, DROPBOX_ROOT)

    vs_id = get_or_create_vector_store()
    attach_vector_store_to_assistant(assistant_id, vs_id)

    uploaded, skipped = 0, 0
    buffer: List[Tuple[str, bytes]] = []

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

        buffer.append((fmeta.name, content))

        if len(buffer) >= 10:
            batch_upload_files_to_vector_store(vs_id, buffer)
            for name, data in buffer:
                manifest[key] = {"hash": h, "last_sync": datetime.utcnow().isoformat()}
                uploaded += 1
            buffer = []

    if buffer:
        batch_upload_files_to_vector_store(vs_id, buffer)
        for name, data in buffer:
            manifest[key] = {"hash": h, "last_sync": datetime.utcnow().isoformat()}
            uploaded += 1

    save_json(MANIFEST_PATH, manifest)
    print(f"✅ Sync complete. Uploaded: {uploaded}, Skipped: {skipped}")
    print(f"Vector store: {vs_id} (saved in {VECTOR_STORE_META_PATH})")

if __name__ == "__main__":
    sync_dropbox_to_assistant()




