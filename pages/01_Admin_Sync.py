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

# ---- Imports from our sync utilities
from scripts.dropbox_sync import (
    sync_dropbox_to_assistant,   # main entry point
    init_dropbox,                # to validate Dropbox auth/root
    resolve_assistant_id,        # to verify Assistant availability
)

# ---- Small helper to capture printed output from the sync so we can show progress
class CapturePrints:
    def __enter__(self):
        self._stdout = sys.stdout
        self.buffer = io.StringIO()
        sys.stdout = self.buffer
        return self
    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._stdout

# ---- Secrets & config helpers
REQUIRED_SECRETS = [
    "OPENAI_API_KEY",
    "DROPBOX_APP_KEY",
    "DROPBOX_APP_SECRET",
    "DROPBOX_REFRESH_TOKEN",
]
OPTIONAL_SECRETS = [
    "ASSISTANT_ID",
    "DROPBOX_ROOT",   # e.g., "/Project_Root"
]

CONFIG_DIR = "config"
MANIFEST_PATH = os.path.join(CONFIG_DIR, "dropbox_manifest.json")
VECTOR_META_PATH = os.path.join(CONFIG_DIR, "vector_store.json")

def has_secret(k: str) -> bool:
    try:
        return bool(st.secrets.get(k))
    except Exception:
        return False

def read_json_if_exists(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

# ---- UI: Preflight status
st.subheader("Preflight")
cols = st.columns(2)
with cols[0]:
    st.markdown("**Secrets**")
    for k in REQUIRED_SECRETS:
        st.write(("✅" if has_secret(k) else "❌"), k)
    for k in OPTIONAL_SECRETS:
        st.write(("ℹ️" if has_secret(k) else "—"), k)

with cols[1]:
    st.markdown("**Config Folder**")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    manifest = read_json_if_exists(MANIFEST_PATH)
    vector_meta = read_json_if_exists(VECTOR_META_PATH)
    st.write("📄", MANIFEST_PATH, "(exists)" if manifest is not None else "(missing)")
    st.write("📄", VECTOR_META_PATH, f"(exists: id={vector_meta.get('vector_store_id')})" if vector_meta else "(missing)")

# ---- Controls
st.subheader("Controls")
override_root = st.text_input("Override Dropbox root (optional)", value="", placeholder="/Project_Root")
batch_size = st.number_input("Batch size", min_value=5, max_value=50, value=10, step=5)
dry_run = st.checkbox("Dry run (list only)", value=False, help="Show what would be synced without uploading (not implemented in script).")

colA, colB = st.columns(2)

# ---- Validate button
with colA:
    if st.button("✅ Validate Setup"):
        with st.spinner("Validating Dropbox auth, root path, and Assistant…"):
            try:
                # 1) Assistant resolution
                a_id = resolve_assistant_id()
                st.success(f"Assistant OK: {a_id}")

                # 2) Dropbox auth + root
                if override_root:
                    os.environ["DROPBOX_ROOT"] = override_root  # the sync module reads env/secrets
                dbx = init_dropbox()  # will raise if creds/root invalid
                root = os.getenv("DROPBOX_ROOT", st.secrets.get("DROPBOX_ROOT", ""))
                st.success(f"Dropbox OK. Root: {root or '(account root)'}")

                # 3) List a handful of files (sanity)
                res = dbx.files_list_folder(root or "", recursive=False)
                names = [e.name for e in res.entries[:10] if hasattr(e, "name")]
                if names:
                    st.write("Found entries:", names)
                else:
                    st.write("No entries found at the specified root (this can still be OK).")

            except Exception as e:
                st.error(f"Validation failed: {e}")
                st.code(traceback.format_exc())

# ---- Sync button
with colB:
    if st.button("🔄 Sync Dropbox → Assistant", type="primary"):
        with st.spinner("Sync in progress…"):
            logs = ""
            ok = False
            try:
                # Support runtime override for root
                if override_root:
                    os.environ["DROPBOX_ROOT"] = override_root
                # (dry_run is not implemented in the sync script; we call live path regardless)
                with CapturePrints() as cap:
                    sync_dropbox_to_assistant(batch_size=int(batch_size))
                logs = cap.buffer.getvalue()
                ok = True
            except Exception as e:
                logs += f"\nERROR: {e}\n{traceback.format_exc()}"
            finally:
                st.text_area("Sync logs", logs or "(no logs)", height=300)
                if ok:
                    st.success("Sync completed. Check config files for manifest & vector store id.")
                else:
                    st.error("Sync failed. See logs above.")

st.divider()
st.caption("Uses Dropbox refresh-token auth and attaches files to the Assistant’s vector store for File Search.")

