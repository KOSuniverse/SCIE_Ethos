import streamlit as st
from datetime import datetime
import mimetypes
from supabase_config import supabase
from supabase_utils import insert_metadata

st.title("🚀 Supabase Test App")

# ---- Test Insert Button ----
if st.button("Insert Test Metadata"):
    metadata = {
        "filename": "test_file.xlsx",
        "folder": "02_Excel",
        "title": "Test File Upload",
        "category": "Demo",
        "tags": ["test", "demo"],
        "summary": "This is a test metadata entry from Streamlit.",
        "filetype": "xlsx",
        "last_modified": datetime.utcnow().isoformat()
    }
    try:
        response = insert_metadata(metadata)
        st.success("✅ Test metadata inserted.")
        st.json(response.data)
    except Exception as e:
        st.error(f"❌ Failed to insert test metadata: {e}")

st.subheader("📦 Supabase Storage Test")

if st.button("List Files in Bucket"):
    try:
        files = supabase.storage.from_("llm-files").list()
        st.success("✅ Files in llm-files bucket:")
        st.json(files)
    except Exception as e:
        st.error(f"❌ Could not access storage bucket: {e}")

# ---- File Upload Section ----
st.markdown("---")
st.subheader("📁 Upload a File to Supabase")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "xlsx", "pptx"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    extension = filename.split(".")[-1].lower()

    # Choose folder based on extension
    folder_map = {
        "xlsx": "02_Excel",
        "docx": "01_Docs",
        "pdf": "01_Docs",
        "pptx": "03_PPT"
    }
    folder = folder_map.get(extension, "unknown")
    file_path = f"{folder}/{filename}"

    try:
        # Upload to Supabase Storage
        supabase.storage.from_("llm-files").upload(file_path, file_bytes)
        st.success(f"✅ Uploaded {filename} to Supabase at `{file_path}`")

        # Generate and insert metadata
        metadata = {
            "filename": filename,
            "folder": folder,
            "title": filename.split(".")[0],
            "category": folder,
            "tags": [extension],
            "summary": f"Uploaded via Streamlit on {datetime.utcnow().isoformat()}",
            "filetype": extension,
            "last_modified": datetime.utcnow().isoformat()
        }
        response = insert_metadata(metadata)
        st.success("✅ Metadata inserted into Supabase.")
        st.json(response.data)

    except Exception as e:
        st.error(f"❌ Failed to upload or insert metadata: {e}")

st.markdown("---")
st.subheader("📋 Supabase Connection Test")

if st.button("Show Metadata Table Rows"):
    try:
        rows = supabase.table("metadata").select("*").limit(5).execute()
        st.success("✅ Retrieved rows from Supabase:")
        st.json(rows.data)
    except Exception as e:
        st.error(f"❌ Failed to query metadata table: {e}")

