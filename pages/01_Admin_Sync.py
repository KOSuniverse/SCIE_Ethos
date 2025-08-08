# pages/01_Admin_Sync.py
import io
import sys
import time
import streamlit as st

st.set_page_config(page_title="Admin: Dropbox Sync", page_icon="🛠️", layout="centered")
st.title("🛠️ Admin — Dropbox → Assistant File Sync")

st.info("This will sync files from Dropbox into the Assistant’s vector store for File Search.")

# We import the function from your script.
# The script already reads st.secrets, so no extra params needed.
from scripts.dropbox_sync import sync_dropbox_to_assistant

# Small helper to capture printed output from the sync so you can see progress in the UI
class CapturePrints:
    def __enter__(self):
        self._stdout = sys.stdout
        self.buffer = io.StringIO()
        sys.stdout = self.buffer
        return self
    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._stdout

if "last_sync_result" not in st.session_state:
    st.session_state.last_sync_result = None

col1, col2 = st.columns(2)
with col1:
    dry_run = st.checkbox("Dry run (list only)", value=False, help="Shows what would be synced without uploading.")
with col2:
    batch_size = st.number_input("Batch size", min_value=5, max_value=50, value=10, step=5)

# Optional: expose a folder override (falls back to secrets[DROPBOX_ROOT])
override_root = st.text_input("Override Dropbox root (optional)", value="", placeholder="/Project_Root")

if st.button("Sync Dropbox → Assistant", type="primary"):
    with st.spinner("Sync in progress..."):
        # If you want to support dry-run & batch_size, you can add kwargs to your sync function.
        # For now we'll just call it as-is. The script reads secrets and env internally.
        # We can temporarily set env overrides here if provided:
        if override_root:
            st.session_state.__dict__["_DROPBOX_ROOT_OVERRIDE"] = override_root  # stash for this run

        # Capture printed progress
        with CapturePrints() as cap:
            try:
                # Optional: monkey-patch env for override (the sync script reads os.environ OR st.secrets)
                import os
                if override_root:
                    os.environ["DROPBOX_ROOT"] = override_root
                # Could also pass batch_size/dry_run if you add support later.
                sync_dropbox_to_assistant()
                ok = True
            except Exception as e:
                ok = False
                st.error(f"Sync failed: {e}")

        logs = cap.buffer.getvalue()
        st.session_state.last_sync_result = logs

        if ok:
            st.success("Sync completed.")
        st.text_area("Sync logs", logs, height=300)

st.divider()
st.caption("Tip: This uses refresh-token auth, so you shouldn’t need to re-generate Dropbox tokens anymore.")
