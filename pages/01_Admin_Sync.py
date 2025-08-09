# pages/01_Admin_Sync.py
import io
import os
import sys
import json
import traceback
import streamlit as st

st.set_page_config(page_title="Admin: Dropbox Sync", page_icon="🛠️", layout="centered")
st.title("🛠️ Admin — Dropbox → Assistant File Sync")
st.info("Sync files from Dropbox into the Assistant’s vector store (File Search).")

# --- Show current OpenAI SDK (helps verify the right version is running)
import openai as _openai
st.caption(f"OpenAI SDK: {_openai.__version__}")

# --- One-time inline upgrade button (useful on Streamlit Cloud if caching blocks requirements)
import subprocess
if st.button("Force-upgrade OpenAI SDK (one-time)"):
    with st.spinner("Upgrading openai…"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "openai==1.56.0", "packaging>=24.1"])
    st.success("Upgraded. Reloading…")
    st.rerun()

# --- Imports from our sync utilities
try:
    from scripts.dropbox_sync import (
        sync_dropbox_to_assistant,   # main entry point
        init_dropbox,                # to validate Dropbox auth/root
        resolve_assistant_id,        # to verify Assistant availability
    )
except Exception as _imp_err:
    st.error(f"Import error: {_imp_err}")
    st.stop()

# --- Capture prints helper (to surface sync progress)
class CapturePrints:
    def __enter__(self):
        self._stdout = sys.stdout
        self.buffer = io.StringIO()
        sys.stdout = self.buffer
        return self
    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._stdout

# --- Secrets & config quick check
REQUIRED_SECRETS = [
    "OPENAI_API_KEY",
    "DROPBOX_APP_KEY",
    "DROPBOX_APP_SECRET",
    "DROPBOX_REFRESH_TOKEN",
]
OPTIONAL_SECRETS = ["ASSISTANT_ID", "DROPBOX_ROOT"]

CONFIG_DIR = "config"
MANIFEST_PATH = os.path.join(CONFIG_DIR, "dropbox_manifest.json")
VECTOR_META_PATH = os.path.join(CONFIG_DIR, "vector_store.json")

def _has_secret(key: str) -> bool:
    try:
        return bool(st.secrets.get(key))
    except Exception:
        return False

def _read_json(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

# --- Preflight
st.subheader("Preflight")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Secrets**")
    for k in REQUIRED_SECRETS:
        st.write(("✅" if _has_secret(k) else "❌"), k)
    for k in OPTIONAL_SECRETS:
        st.write(("ℹ️" if _has_secret(k) else "—"), k)
with c2:
    st.markdown("**Config Folder**")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    mf = _read_json(MANIFEST_PATH)
    vm = _read_json(VECTOR_META_PATH)
    st.write("📄", MANIFEST_PATH, "(exists)" if mf is not None else "(missing)")
    st.write("📄", VECTOR_META_PATH, f"(exists: id={vm.get('vector_store_id')})" if vm else "(missing)")

# --- Controls
st.subheader("Controls")
override_root = st.text_input("Override Dropbox root (optional)", value="", placeholder="/Project_Root")
batch_size = st.number_input("Batch size", min_value=5, max_value=50, value=10, step=5)
dry_run = st.checkbox("Dry run (list only)", value=False, help="Show what would be synced without uploading (not implemented in script).")

left, right = st.columns(2)

# --- Validate
with left:
    if st.button("✅ Validate Setup"):
        with st.spinner("Validating Assistant + Dropbox…"):
            try:
                # Assistant
                a_id = resolve_assistant_id()
                st.success(f"Assistant OK: {a_id}")

                # Dropbox auth + root
                if override_root:
                    os.environ["DROPBOX_ROOT"] = override_root
                dbx = init_dropbox()  # raises if creds/root invalid
                root = os.getenv("DROPBOX_ROOT", st.secrets.get("DROPBOX_ROOT", ""))
                st.success(f"Dropbox OK. Root: {root or '(account root)'}")

                # List a few entries
                res = dbx.files_list_folder(root or "", recursive=False)
                names = [e.name for e in res.entries[:10] if hasattr(e, "name")]
                st.write("Found entries:" if names else "No entries at root.", names or [])
            except Exception as e:
                st.error(f"Validation failed: {e}")
                st.code(traceback.format_exc())

# --- Sync
with right:
    if st.button("🔄 Sync Dropbox → Assistant", type="primary"):
        with st.spinner("Sync in progress…"):
            logs = ""
            ok = False
            try:
                if override_root:
                    os.environ["DROPBOX_ROOT"] = override_root
                if dry_run:
                    st.warning("Dry run not implemented; running live sync.")
                with CapturePrints() as cap:
                    sync_dropbox_to_assistant(batch_size=int(batch_size))
                logs = cap.buffer.getvalue()
                ok = True
            except Exception as e:
                logs += f"\nERROR: {e}\n{traceback.format_exc()}"
            finally:
                st.text_area("Sync logs", logs or "(no logs)", height=300)
                if ok:
                    st.success("Sync completed. Check /config for manifest & vector store id.")
                else:
                    st.error("Sync failed. See logs above.")

st.divider()
st.caption("Uses Dropbox refresh-token auth and attaches files to the Assistant’s vector store for File Search.")


