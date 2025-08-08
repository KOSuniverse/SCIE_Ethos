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
    # fail fast on bad token/path
    dbx.files_list_folder(DROPBOX_ROOT, recursive=False)
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

# --- Vector store helpers ---
def get_or_create_vector_store() -> str:
    meta = load_json(VECTOR_STORE_META_PATH, {})
    vs_id = meta.get("vector_store_id")

    if vs_id:
        # sanity check it exists
        try:
            client.beta.vector_stores.retrieve(vs_id)
            return vs_id
        except Exception:
            pass  # recreate if missing

    vs = client.beta.vector_stores.create(name="SCIE Ethos Source Docs")
    save_json(VECTOR_STORE_META_PATH, {"vector_store_id": vs.id})
    return vs.id

def attach_vector_store_to_assistant(vs_id: str):
    # Retrieve current assistant to avoid nuking other tool settings
    a = client.beta.assistants.retrieve(assistant_id)
    tr = a.tool_resources or {}

    # Merge this vector store ID with any existing ones
    existing_ids = []
    try:
        existing_ids = tr.get("file_search", {}).get("vector_store_ids", [])
    except Exception:
        pass

    # Avoid duplicates
    if vs_id not in existing_ids:
        new_ids = existing_ids + [vs_id]
        client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={
                "file_search": {"vector_store_ids": new_ids}
            },
            tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
        )

def batch_upload_files_to_vector_store(vs_id: str, file_tuples: List[Tuple[str, bytes]]):
    """
    file_tuples: list of (filename, content_bytes)
    Writes to temp files to satisfy SDK, then upload as a batch and wait.
    """
    temp_paths = []
    try:
        for name, data in file_tuples:
            fd, p = tempfile.mkstemp()
            with os.fdopen(fd, "wb") as w:
                w.write(data)
            # rename to keep original extension for better parsing
            base, ext = os.path.splitext(name)
            final_p = p + ext
            os.rename(p, final_p)
            temp_paths.append(final_p)

        with client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs_id,
            files=[open(p, "rb") for p in temp_paths],
        ) as batch:  # context manager ensures file handles close
            # .upload_and_poll returns a FileBatch; wait until it completes internally
            pass
    finally:
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
    upload_buffer: List[Tuple[str, bytes]] = []

    for fmeta in dropbox_files:
        ext = os.path.splitext(fmeta.name)[1].lower()
        if ext not in ALLOWED_EXTS:
            continue

        _, resp = dbx.files_download(fmeta.path_lower)
        content = resp.content
        h = file_hash(content)

        if fmeta.path_lower in manifest and manifest[fmeta.path_lower]["hash"] == h:
            skipped += 1
            continue

        upload_buffer.append((fmeta.name, content))

        # Upload in chunks to avoid huge batches; tune as needed
        if len(upload_buffer) >= 10:
            batch_upload_files_to_vector_store(vs_id, upload_buffer)
            for name, data in upload_buffer:
                manifest[f"{DROPBOX_ROOT}/{name}".replace("//", "/").lower()] = {
                    "hash": file_hash(data),
                    "last_sync": datetime.utcnow().isoformat(),
                }
                uploaded += 1
            upload_buffer = []

    # Flush remaining
    if upload_buffer:
        batch_upload_files_to_vector_store(vs_id, upload_buffer)
        for name, data in upload_buffer:
            manifest[f"{DROPBOX_ROOT}/{name}".replace("//", "/").lower()] = {
                "hash": file_hash(data),
                "last_sync": datetime.utcnow().isoformat(),
            }
            uploaded += 1

    save_json(MANIFEST_PATH, manifest)
    print(f"✅ Sync complete. Uploaded: {uploaded}, Skipped: {skipped}")
    print(f"Vector store: {vs_id} (saved in {VECTOR_STORE_META_PATH})")

if __name__ == "__main__":
    sync_dropbox_to_assistant()


