import streamlit as st
from supabase_utils import insert_metadata
from datetime import datetime

st.title("🚀 Supabase Test App")

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
   import mimetypes
from supabase_config import supabase
from supabase_utils import insert_metadata
import base64
import io
from datetime import datetime
import streamlit as st

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

    # Upload to Supabase Storage
    file_path = f"{folder}/{filename}"
    try:
        supabase.storage.from_("llm-files").upload(file_path, file_bytes, {"content-type": mimetypes.guess_type(filename)[0]})
        st.success(f"✅ Uploaded {filename} to Supabase at `{file_path}`")

        # Insert metadata
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
 
    try:
        response = insert_metadata(metadata)
        st.success("✅ Metadata inserted into Supabase.")
        st.json(response.data)
    except Exception as e:
        st.error(f"❌ Error inserting metadata: {e}")
