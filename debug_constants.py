import sys
import os
sys.path.append('PY Files')

print("=== DEBUGGING CONSTANTS ===")
print("DROPBOX_ROOT env:", repr(os.getenv("DROPBOX_ROOT")))
print("DROPBOX_ROOT env with default:", repr(os.getenv("DROPBOX_ROOT", "")))

# Try to replicate constants logic
try:
    import streamlit as st
    print("Streamlit imported successfully")
    _DBX = st.secrets.get("DROPBOX_ROOT", os.getenv("DROPBOX_ROOT", "")).strip("/")
    print("Using streamlit path")
except Exception as e:
    print(f"Streamlit exception: {e}")
    _DBX = os.getenv("DROPBOX_ROOT", "").strip("/")
    print("Using environment path")

print("_DBX value:", repr(_DBX))
print("_DBX truthiness:", bool(_DBX))

PROJECT_ROOT = f"/{_DBX}" if _DBX else "/project_root"
print("PROJECT_ROOT:", repr(PROJECT_ROOT))

# Now import actual constants
from constants import PROJECT_ROOT as ACTUAL_PROJECT_ROOT
print("ACTUAL PROJECT_ROOT:", repr(ACTUAL_PROJECT_ROOT))
