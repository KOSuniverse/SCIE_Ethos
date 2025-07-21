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
    try:
        response = insert_metadata(metadata)
        st.success("✅ Metadata inserted into Supabase.")
        st.json(response.data)
    except Exception as e:
        st.error(f"❌ Error inserting metadata: {e}")
