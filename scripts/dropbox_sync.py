# scripts/dropbox_sync.py
import os, json, hashlib, tempfile
from datetime import datetime
from typing import List, Tuple

import dropbox
from openai import OpenAI

# --- Resolve repo root regardless of CWD ---
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(HERE, ".."))

# --- CONFIG (env-configurable) ---
DROPBOX_ROOT = os.environ.get("DROPBOX_ROOT", "")  # e.g., "/Project_Root" or "" for root
ALLOWED_EXTS = {".pdf", ".docx", ".xlsx", ".csv", ".txt"}
MANIFEST_PATH = os.path.join(REPO_DIR, "config", "dropbox_manifest.json")
ASSISTANT_META_PATH = os.path.join(REPO_DIR, "config", "assistant.json")
VECTOR_STORE_META_PATH = os.path.join(REPO_DIR, "config", "vector_store.json")

# --- ENV ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")
ASSISTANT_ID_ENV = os.getenv("ASSISTANT_ID")  # optional fallback

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set")
if not DROPBOX_TOKEN:
    raise EnvironmentError("DROPBOX_TOKEN not set")

client = OpenAI()

# --- Resolve assistant_id from env first, else file ---
def resolve_assistant_id():
    if ASSISTANT_ID_ENV:
        return ASSISTANT_ID_ENV
    if os.path.exists(ASSISTANT_META_PATH):
        with open(ASSISTANT_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["assistant_id"]
    raise FileNotFoundError(
        f"No assistant ID found. Set ASSISTANT_ID env var or provide {ASSISTANT_META_PATH}."
    )

assistant_id = resolve_assistant_id()

# --- Dropbox init + probe ---
def init_dropbox():
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    dbx.files_list_folder(DROPBOX_ROOT, recursive=False)  # fail fast
    return dbx

dbx = init_dropbox()

# --- Helpers ---
def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def list_dropbox_files(path):
    files = []
    result = dbx.files_list_folder(path, recursive=True)
    files.extend(result.entries)
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        files.extend(result.entries)
    return [f for f in files if isinstance(f, dropbox.files.FileMetadata)]

# --- Vector store helpers (new API in 1.99.x) ---
def get_or_create_vector_store() -> str:
    meta = load_json(VECTOR_STORE_META_PATH, {})
    vs_id = meta.get("vector_store_id")

    if vs_id:
        try:
            client.vector_stores.retrieve(vs_id)
            return vs_id
        except Exception:
            pass  # recreate if it was deleted

    vs = client.vector_stores.create(name="SCIE Ethos Source Docs")
    save_json(VECTOR_STORE_META_PATH, {"vector_store_id": vs.id})
    return vs.id

def attach_vector_store_to_assistant(vs_id: str):
    # Fetch assistant to preserve existing tool resources
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
    """
    file_tuples: list of (filename, content_bytes)
    Writes to temp files to satisfy SDK; uploads as a batch and blocks until complete.
    """
    temp_paths = []
    handles = []
    try:
        for name, data in file_tuples:
            fd, p = tempfile.mkstemp()
            with os.fdopen(fd, "wb") as w:
                w.write(data)
            base, ext = os.path.splitext(name)
            final_p = p + ext
            os.rename(p, final_p)
            temp_paths.append(final_p)

        # Open all files
        for p in temp_paths:
            handles.append(open(p, "rb"))

        # New API: not under beta
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs_id,
            files=handles,
        )
        # batch.status should be "completed" here
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

# --- Main sync ---
def sync_dropbox_to_assistant():
    manifest = load_json(MANIFEST_PATH, {})
    dropbox_files = list_dropbox_files(DROPBOX_ROOT)

    vs_id = get_or_create_vector_store()
    attach_vector_store_to_assistant(vs_id)

    uploaded, skipped = 0, 0
    buffer: List[Tuple[str, bytes]] = []

    for fmeta in dropbox_files:
        ext = os.path.splitext(fmeta.name)[1].lower()
        if ext not in ALLOWED_EXTS:
            continue

        _, resp = dbx.files_download(fmeta.path_lower)
        content = resp.content
        h = file_hash(content)

        key = fmeta.path_lower  # stable key in manifest
        if key in manifest and manifest[key]["hash"] == h:
            skipped += 1
            continue

        buffer.append((fmeta.name, content))

        if len(buffer) >= 10:  # tune batch size as you like
            batch_upload_files_to_vector_store(vs_id, buffer)
            for name, data in buffer:
                # store by dropbox path if available (here we just use name)
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



