import streamlit as st
from datetime import datetime
import mimetypes
from supabase_config import supabase
from supabase_utils import insert_metadata
import pandas as pd
import io

def extract_text_from_excel(file_bytes):
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        text = ""
        for sheet_name, sheet_df in df.items():
            text += f"\n\nSheet: {sheet_name}\n"
            text += sheet_df.astype(str).to_string(index=False)
        return text[:6000]  # limit to ~6K characters for GPT
    except Exception as e:
        return f"Error reading Excel: {e}"

import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_metadata_from_text(text):
    prompt = f"""
You are a metadata extraction assistant.

Given the following document content, extract:
- A short, descriptive title
- A one-word category
- A 1–2 sentence summary
- A list of 3–5 relevant tags (as a Python list)

Respond with only a valid Python dictionary, no extra text or comments.

Document content:
{text}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured metadata."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        reply = response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ GPT request failed: {e}")
        return {
            "title": "Untitled",
            "category": "Unknown",
            "summary": "Metadata generation failed.",
            "tags": []
        }

    st.subheader("🧠 GPT Raw Reply:")
    st.code(reply)

    try:
        parsed = eval(reply)
        assert isinstance(parsed, dict), "Reply is not a dictionary"
        return parsed
    except Exception as parse_error:
        st.error(f"⚠️ Failed to parse GPT reply: {parse_error}")
        return {
            "title": "Untitled",
            "category": "Unknown",
            "summary": "Metadata generation failed.",
            "tags": []
        }

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

        # Extract text and generate AI metadata
        if extension == "xlsx":
            text = extract_text_from_excel(file_bytes)
        else:
            text = ""  # You can add other extractors for docx, pdf, pptx if needed

        st.subheader("🧾 Extracted Text from Excel:")
        st.code(text[:1000])  # preview only first 1000 chars

        ai_metadata, gpt_reply = generate_metadata_from_text(text), ""

        st.subheader("🧠 GPT Raw Reply")
        st.code(gpt_reply)

        metadata = {
            "filename": filename,
            "folder": folder,
            "title": ai_metadata["title"],
            "category": ai_metadata["category"],
            "tags": ai_metadata["tags"],
            "summary": ai_metadata["summary"],
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

