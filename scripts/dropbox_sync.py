# scripts/dropbox_sync.py
import os, json, hashlib, tempfile, mimetypes
from datetime import datetime
from typing import List, Tuple

# --- SDK version info (optional caption shown in UI separately) ---
import openai as _openai
try:
    from packaging import version as _v
except Exception:
    _v = None

MIN_VER = "1.52.0"
_current = getattr(_openai, "__version__", "0")

def _too_old(vstr: str, minv: str) -> bool:
    if _v:
        try:
            return _v.parse(vstr) < _v.parse(minv)
        except Exception:
            pass
    def _parts(s): return [int(p) for p in s.split(".") if p.replace(".", "").isdigit()]
    return _parts(vstr) < _parts(minv)

if _too_old(_current, MIN_VER):
    raise RuntimeError(
        f"OpenAI SDK too old ({_current}). Please set openai>={MIN_VER} in requirements.txt and redeploy."
    )

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

# ---------------- Vector Store adapter (handles SDK naming changes) ----------------
class _VS:
    """
    Thin adapter that finds the vector store namespace across SDK variants:
    - client.beta.vector_stores
    - client.beta.vectorstores
    - client.vector_stores
    - client.vectorstores
    And exposes create/retrieve/file_batches.upload_and_poll().
    """
    def __init__(self, client: OpenAI):
        self.client = client
        self.ns = None
        # try candidates in order
        cand = [
            getattr(getattr(client, "beta", object()), "vector_stores", None),
            getattr(getattr(client, "beta", object()), "vectorstores", None),
            getattr(client, "vector_stores", None),
            getattr(client, "vectorstores", None),
        ]
        for c in cand:
            if c is not None:
                self.ns = c
                break
        if self.ns is None:
            raise AttributeError(
                "OpenAI vector store API not found on this SDK. "
                "Tried beta.vector_stores/vectorstores and root vector_stores/vectorstores."
            )

        # file_batches namespace (snake_case / variants)
        self.file_batches = getattr(self.ns, "file_batches", None) or getattr(self.ns, "fileBatches", None)
        if self.file_batches is None:
            # Some SDKs may attach upload APIs differently in future; fail clearly if missing
            raise AttributeError("Vector store 'file_batches' API not found on this SDK.")

    def create(self, **kwargs):
        return self.ns.create(**kwargs)

    def retrieve(self, vector_store_id: str):
        return self.ns.retrieve(vector_store_id)

    def upload_batch_and_poll(self, vector_store_id: str, files):
        # Standard path (1.5x–1.9x)
        if hasattr(self.file_batches, "upload_and_poll"):
            return self.file_batches.upload_and_poll(vector_store_id=vector_store_id, files=files)
        # Fallback: try 'upload' + manual poll if present (future-proof)
        if hasattr(self.file_batches, "upload"):
            batch = self.file_batches.upload(vector_store_id=vector_store_id, files=files)
            # If the returned object has .status=='completed', just return it; otherwise best-effort
            return getattr(batch, "status", "completed")
        raise AttributeError("No compatible upload method found on vector store file_batches.")

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

# --- Vector store helpers ---
def get_or_create_vector_store() -> str:
    vs_meta = load_json(VECTOR_STORE_META_PATH, {})
    vs_id = vs_meta.get("vector_store_id")
    vs = _VS(client)

    if vs_id:
        try:
            vs.retrieve(vs_id)
            return vs_id
        except Exception:
            pass  # recreate if deleted

    created = vs.create(name="SCIE Ethos Source Docs")
    # Some SDKs return object with .id; others dict-like; handle both
    new_id = getattr(created, "id", None) or created.get("id")
    if not new_id:
        raise RuntimeError("Could not obtain vector_store.id from create().")
    save_json(VECTOR_STORE_META_PATH, {"vector_store_id": new_id})
    return new_id

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
    import io as _io
    files = []
    for name, data in file_tuples:
        mime, _ = mimetypes.guess_type(name)
        files.append((name, _io.BytesIO(data), mime or "application/octet-stream"))

    vs = _VS(client)
    status = vs.upload_batch_and_poll(vector_store_id=vs_id, files=files)
    return status  # often "completed"

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





