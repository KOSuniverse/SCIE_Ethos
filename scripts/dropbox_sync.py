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
    
    Architecture principle: "Documents use OpenAI File Search, Excel/CSV use Code Interpreter"
    SOLUTION: PDFs/DOCs go to vector store, Excel/CSV stay in Dropbox for Code Interpreter
    """
    # Normalize path for comparison
    path_lower = file_path.lower().replace("\\", "/")
    
    # Exclude by folder location (generated artifacts)
    for exclude_folder in EXCLUDE_FOLDERS:
        if exclude_folder in path_lower:
            return True, f"Generated artifact folder: {exclude_folder}"
    
    # Exclude tabular files (OpenAI File Search doesn't support them)
    # These will be available via Code Interpreter from Dropbox
    if ext in TABULAR_EXTS:
        return True, "Tabular file (available via Code Interpreter from Dropbox)"
    
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

# Assistant configuration - can be created automatically or use existing
ASSISTANT_ID = SECRETS.get("ASSISTANT_ID")  # optional; can fall back to assistant.json if you keep one in Dropbox later
AUTO_CREATE_ASSISTANT = SECRETS.get("AUTO_CREATE_ASSISTANT", "true").lower() == "true"

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
def create_assistant() -> str:
    """Create a new OpenAI Assistant with File Search + Code Interpreter."""
    try:
        # Load instructions from prompts/instructions_master.yaml
        instructions_path = "prompts/instructions_master.yaml"
        if os.path.exists(instructions_path):
            import yaml
            with open(instructions_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Convert YAML to instruction text
            instructions_text = _yaml_to_instructions(config)
        else:
            # Fallback instructions
            instructions_text = """You are SCIE Ethos — an enterprise supply-chain analyst copilot.
You must produce accurate, auditable, and action-oriented answers grounded in data and cited sources.
Use File Search to find relevant documents and Code Interpreter to analyze Excel/CSV data.
Always cite your sources and separate findings from data gaps."""
        
        # Create the assistant
        assistant = client.beta.assistants.create(
            name="SCIE Ethos Supply Chain & Inventory Analyst",
            description="Enterprise supply-chain analyst copilot with File Search + Code Interpreter",
            model=OPENAI_MODEL,
            instructions=instructions_text,
            tools=[{"type": "file_search"}, {"type": "code_interpreter"}]
        )
        
        # Save assistant metadata
        assistant_meta = {
            "assistant_id": assistant.id,
            "name": assistant.name,
            "model": assistant.model,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Save to local file
        os.makedirs("prompts", exist_ok=True)
        with open("prompts/assistant.json", "w", encoding="utf-8") as f:
            json.dump(assistant_meta, f, indent=2)
        
        # Also save to Dropbox for cloud access
        dbx_write_json(f"{DBX_CONFIG_DIR}/assistant.json", assistant_meta)
        
        print(f"✅ Created new Assistant: {assistant.id}")
        return assistant.id
        
    except Exception as e:
        print(f"❌ Failed to create assistant: {e}")
        raise

def _yaml_to_instructions(config: dict) -> str:
    """Convert YAML config to instruction text for the Assistant."""
    lines = []
    
    # Assistant profile
    ap = config.get("assistant_profile", {})
    if ap.get("name") and ap.get("description"):
        lines.append(f"You are {ap['name']}: {ap['description']}")
    
    # Core directives
    directives = config.get("core_directives", [])
    if directives:
        lines.append("\nCore Directives:")
        for directive in directives:
            lines.append(f"- {directive}")
    
    # Intents
    intents = config.get("intents", {})
    if intents:
        lines.append("\nIntents & Sub-skills:")
        for intent, spec in intents.items():
            desc = spec.get("description", "")
            subs = spec.get("subskills", [])
            lines.append(f"- {intent}: {desc}")
            if subs:
                lines.append(f"  * Sub-skills: {', '.join(subs)}")
    
    # Gap detection
    gap_rules = config.get("gap_detection_rules", [])
    if gap_rules:
        lines.append("\nGap Detection Rules:")
        for rule in gap_rules:
            lines.append(f"- {rule}")
    
    # Output templates
    templates = config.get("output_templates", {})
    if templates:
        lines.append("\nOutput Templates:")
        for name, template in templates.items():
            lines.append(f"\n{name}:")
            lines.append(template)
    
    lines.append("\nAlways ground answers in retrieved files, cite sources, and separate 'Data Needed' from findings.")
    return "\n".join(lines)

def resolve_assistant_id() -> str:
    """Resolve the assistant ID, creating one if needed."""
    if ASSISTANT_ID:
        return ASSISTANT_ID
    
    # Try to load from local file
    try:
        with open("prompts/assistant.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            return meta["assistant_id"]
    except FileNotFoundError:
        pass
    
    # Try to load from Dropbox
    try:
        meta = dbx_read_json(f"{DBX_CONFIG_DIR}/assistant.json")
        if meta and meta.get("assistant_id"):
            return meta["assistant_id"]
    except Exception:
        pass
    
    # Create new assistant if auto-creation is enabled
    if AUTO_CREATE_ASSISTANT:
        return create_assistant()
    
    raise FileNotFoundError("ASSISTANT_ID not set and auto-creation disabled. Set ASSISTANT_ID in secrets or enable AUTO_CREATE_ASSISTANT.")

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
def sync_dropbox_to_assistant(batch_size: int = 10, specific_folders: List[str] = None):
    """
    Sync Dropbox files to OpenAI Assistant.
    
    Args:
        batch_size: Number of files to upload in each batch
        specific_folders: List of specific folders to sync (e.g., ["04_Data/00_Raw_Files", "06_LLM_Knowledge_Base"])
    """
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
    print(f"   📤 Uploaded to OpenAI File Search: {uploaded}")
    print(f"   ⏭️  Skipped (unchanged): {skipped}")
    print(f"   📁 Excluded (artifacts): {excluded_artifacts}")
    print(f"   📊 Excel/CSV files (Code Interpreter): {excluded_tabular}")
    print(f"   ❌ Excluded (unsupported): {unsupported}")
    print(f"   🔗 Vector store: {vs_id}")
    print(f"   💾 Manifest saved: {DBX_MANIFEST_PATH}")
    print(f"   📋 Total files in Dropbox: {len(dropbox_files)}")
    print(f"   🔍 Files available for File Search: {uploaded + skipped}")
    print(f"   📈 Files available for Code Interpreter: {excluded_tabular}")

def sync_by_path_contract():
    """Sync files according to the path contract structure."""
    print("🔄 Syncing according to path contract...")
    
    # Define folders to sync based on path_contract.yaml
    sync_folders = [
        "04_Data/00_Raw_Files",           # Raw data files
        "06_LLM_Knowledge_Base",          # Knowledge base documents
        "04_Data/01_Cleansed_Files",      # Cleaned data files
        "04_Data/04_Metadata",            # Metadata files
    ]
    
    for folder in sync_folders:
        print(f"\n📁 Syncing folder: {folder}")
        try:
            sync_dropbox_to_assistant(batch_size=5, specific_folders=[folder])
        except Exception as e:
            print(f"❌ Failed to sync {folder}: {e}")
            continue
    
    print("\n✅ Path contract sync complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "path-contract":
        sync_by_path_contract()
    else:
        sync_dropbox_to_assistant()






