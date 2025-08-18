# pages/01_Admin_Sync.py
import io, os, sys, json, traceback, subprocess
import streamlit as st
import dropbox

st.set_page_config(page_title="Admin: Dropbox Sync", page_icon="🛠️", layout="centered")
st.title("🛠️ Admin — Dropbox → Assistant File Sync")
st.info("Sync files from Dropbox into the Assistant’s vector store (File Search).")

# Show OpenAI SDK
import openai as _openai
st.caption(f"OpenAI SDK: {_openai.__version__}")

# One‑time SDK upgrade (optional)
if st.button("Force-upgrade OpenAI SDK (one-time)"):
    with st.spinner("Upgrading openai…"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "openai==1.56.0", "packaging>=24.1"])
    st.success("Upgraded. Reloading…"); st.rerun()

# Sync utilities
try:
    from scripts.dropbox_sync import sync_dropbox_to_assistant, init_dropbox, resolve_assistant_id
except Exception as _imp_err:
    st.error(f"Import error: {_imp_err}"); st.stop()

class CapturePrints:
    def __enter__(self): self._stdout=sys.stdout; self.buffer=io.StringIO(); sys.stdout=self.buffer; return self
    def __exit__(self, t, e, tb): sys.stdout=self._stdout

REQUIRED_SECRETS = ["OPENAI_API_KEY","DROPBOX_APP_KEY","DROPBOX_APP_SECRET","DROPBOX_REFRESH_TOKEN"]
OPTIONAL_SECRETS  = ["ASSISTANT_ID","DROPBOX_ROOT"]

# Paths MUST be app‑root relative for Dropbox SDK (no "/Apps/Ethos LLM" prefix)
root = st.secrets.get("DROPBOX_ROOT", "").strip("/")
CONFIG_DIR = f"/{root}/config" if root else "/config"
MANIFEST_PATH = os.path.join(CONFIG_DIR, "dropbox_manifest.json")
VECTOR_META_PATH = os.path.join(CONFIG_DIR, "vector_store.json")

def _has_secret(k:str)->bool:
    try: return bool(st.secrets.get(k))
    except Exception: return False

def _read_json(path:str):
    try:
        dbx = init_dropbox()
        _, res = dbx.files_download(path)
        return json.loads(res.content.decode("utf-8"))
    except Exception:
        return None

def _ensure_dropbox_folder(dbx: dropbox.Dropbox, folder:str):
    folder = (folder or "/").rstrip("/") or "/"
    if folder == "/": return
    try:
        dbx.files_get_metadata(folder)
    except dropbox.exceptions.ApiError:
        dbx.files_create_folder_v2(folder, autorename=False)

def _write_json(path:str, data:dict):
    dbx = init_dropbox()
    parent = os.path.dirname(path.rstrip("/")) or "/"
    _ensure_dropbox_folder(dbx, parent)
    dbx.files_upload(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                     path, mode=dropbox.files.WriteMode.overwrite, mute=True)

# --- Preflight
st.subheader("Preflight")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Secrets**")
    for k in REQUIRED_SECRETS: st.write(("✅" if _has_secret(k) else "❌"), k)
    for k in OPTIONAL_SECRETS: st.write(("ℹ️" if _has_secret(k) else "—"), k)

with c2:
    st.markdown("**Config Folder**")
    mf = _read_json(MANIFEST_PATH)
    vm = _read_json(VECTOR_META_PATH)

    # One‑time migration if previous runs wrote local files
    local_cfg = os.path.join(os.getcwd(), "config")
    local_manifest = os.path.join(local_cfg, "dropbox_manifest.json")
    local_vector  = os.path.join(local_cfg, "vector_store.json")
    if vm is None and os.path.exists(local_vector):
        _write_json(VECTOR_META_PATH, json.load(open(local_vector, "r", encoding="utf-8")))
        vm = _read_json(VECTOR_META_PATH)
    if mf is None and os.path.exists(local_manifest):
        _write_json(MANIFEST_PATH, json.load(open(local_manifest, "r", encoding="utf-8")))
        mf = _read_json(MANIFEST_PATH)

    st.write("📄", MANIFEST_PATH, "(exists)" if mf is not None else "(missing)")
    st.write("📄", VECTOR_META_PATH, f"(exists: id={vm.get('vector_store_id')})" if vm else "(missing)")

# --- Controls
st.subheader("Controls")
override_root = st.text_input("Override Dropbox root (optional)", value="", placeholder="/Project_Root")
batch_size    = st.number_input("Batch size", min_value=5, max_value=50, value=10, step=5)
dry_run       = st.checkbox("Dry run (list only)", value=False, help="Show what would be synced without uploading (not implemented).")

left, right = st.columns(2)

# --- Validate
with left:
    if st.button("✅ Validate Setup"):
        with st.spinner("Validating Assistant + Dropbox…"):
            try:
                a_id = resolve_assistant_id(); st.success(f"Assistant OK: {a_id}")
                if override_root: os.environ["DROPBOX_ROOT"] = override_root
                dbx = init_dropbox()
                # Use the corrected PROJECT_ROOT from constants instead of raw env var
                from constants import PROJECT_ROOT
                root_to_test = PROJECT_ROOT
                st.success(f"Dropbox OK. Root: {root_to_test}")
                res = dbx.files_list_folder(root_to_test, recursive=False)
                names = [e.name for e in res.entries[:10] if hasattr(e, "name")]
                st.write("Found entries:" if names else "No entries at root.", names or [])
            except Exception as e:
                st.error(f"Validation failed: {e}"); st.code(traceback.format_exc())

# --- Sync
with right:
    if st.button("🔄 Sync Dropbox → Assistant", type="primary"):
        with st.spinner("Sync in progress…"):
            logs, ok = "", False
            try:
                if override_root: os.environ["DROPBOX_ROOT"] = override_root
                if dry_run: st.warning("Dry run not implemented; running live sync.")
                with CapturePrints() as cap:
                    sync_dropbox_to_assistant(batch_size=int(batch_size))

                logs = cap.buffer.getvalue(); ok = True
            except Exception as e:
                logs += f"\nERROR: {e}\n{traceback.format_exc()}"
            finally:
                st.text_area("Sync logs", logs or "(no logs)", height=300)
                st.success("Sync completed. Check /config for manifest & vector store id.") if ok else st.error("Sync failed. See logs above.")

st.divider()
st.caption("Uses Dropbox refresh‑token auth and attaches files to the Assistant’s vector store for File Search.")




