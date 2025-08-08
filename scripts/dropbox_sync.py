# scripts/dropbox_sync.py
import os
import json
import dropbox
import hashlib
from datetime import datetime
from openai import OpenAI

# --- CONFIG ---
DROPBOX_ROOT = ""   # e.g., "/SCIE_Ethos_Source" or "" for root
ALLOWED_EXTS = {".pdf", ".docx", ".xlsx", ".csv", ".txt"}
MANIFEST_PATH = "config/dropbox_manifest.json"
ASSISTANT_META_PATH = "config/assistant.json"

# --- INIT ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set")
if not DROPBOX_TOKEN:
    raise EnvironmentError("DROPBOX_TOKEN not set")

client = OpenAI()
dbx = dropbox.Dropbox(DROPBOX_TOKEN)

# --- HELPERS ---
def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(data):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def list_dropbox_files(path):
    files = []
    result = dbx.files_list_folder(path, recursive=True)
    files.extend(result.entries)
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        files.extend(result.entries)
    return [f for f in files if isinstance(f, dropbox.files.FileMetadata)]

def upload_to_assistant(file_name, file_bytes, assistant_id):
    uploaded_file = client.files.create(
        file=(file_name, file_bytes),
        purpose="assistants"
    )
    client.beta.assistants.files.create(
        assistant_id=assistant_id,
        file_id=uploaded_file.id
    )
    return uploaded_file.id

# --- MAIN SYNC ---
def sync_dropbox_to_assistant():
    with open(ASSISTANT_META_PATH, "r", encoding="utf-8") as f:
        assistant_meta = json.load(f)
    assistant_id = assistant_meta["assistant_id"]

    manifest = load_manifest()
    dropbox_files = list_dropbox_files(DROPBOX_ROOT)

    uploaded_count = 0
    skipped_count = 0

    for fmeta in dropbox_files:
        ext = os.path.splitext(fmeta.name)[1].lower()
        if ext not in ALLOWED_EXTS:
            continue

        # Download
        _, resp = dbx.files_download(fmeta.path_lower)
        content = resp.content
        h = file_hash(content)

        # Skip if unchanged
        if fmeta.path_lower in manifest and manifest[fmeta.path_lower]["hash"] == h:
            skipped_count += 1
            continue

        # Upload to Assistant
        file_id = upload_to_assistant(fmeta.name, content, assistant_id)

        manifest[fmeta.path_lower] = {
            "hash": h,
            "last_sync": datetime.utcnow().isoformat(),
            "file_id": file_id
        }
        uploaded_count += 1
        print(f"Uploaded: {fmeta.name}")

    save_manifest(manifest)
    print(f"✅ Sync complete. Uploaded: {uploaded_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    sync_dropbox_to_assistant()
