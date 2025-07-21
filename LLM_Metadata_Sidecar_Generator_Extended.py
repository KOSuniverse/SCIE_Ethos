"""
LLM Metadata Sidecar Generator Script (Extended)
-------------------------------------------------
This script scans a project directory recursively for supported document types,
extracts structured metadata from the document itself (if present),
or falls back to OpenAI GPT to generate metadata.
It then saves a sidecar .json file into a `_metadata/` subfolder next to each file.

Supported: .docx, .xlsx, .pptx, .pdf
"""

import json
from datetime import datetime
from docx import Document
import openpyxl
from pptx import Presentation
import pdfplumber
from openai import OpenAI
import numpy as np
import pandas as pd
import streamlit as st
from supabase_utils import list_supported_files, upload_file, download_file, save_metadata, load_metadata, load_global_aliases, update_global_aliases

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

def enrich_tags(existing_tags, query_log):
    new_tags = set(existing_tags)
    for q in query_log.get("most_common_queries", []):
        for word in q.lower().split():
            if word not in new_tags and len(word) > 3:
                new_tags.add(word)
    return list(new_tags)

def map_columns_to_concepts(columns, global_aliases=None):
    unmapped = [col for col in columns if not global_aliases or col not in global_aliases]
    mapping = global_aliases.copy() if global_aliases else {}
    if unmapped:
        prompt = (
            "Map the following Excel column headers to standard business concepts. "
            "Return a JSON dictionary where each key is the original column name and each value is a standard concept "
            "(e.g., 'quantity', 'part_number', 'location'). If a header is junk or unrecognizable, map it to 'ignore'.\n\n"
            f"Columns: {unmapped}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        try:
            new_mapping = json.loads(raw)
            mapping.update(new_mapping)
        except Exception:
            pass
    return mapping

def profile_excel(file_path):
    wb = openpyxl.load_workbook(file_path, read_only=True)
    profile = {}
    for sheet in wb.worksheets:
        sheet_profile = {}
        df = pd.DataFrame(sheet.values)
        df.columns = df.iloc[0]
        df = df[1:]
        for col in df.columns:
            col_data = df[col]
            sheet_profile[str(col)] = {
                "type": str(col_data.dtype),
                "null_pct": float(col_data.isnull().mean()),
                "min": col_data.min() if pd.api.types.is_numeric_dtype(col_data) else None,
                "max": col_data.max() if pd.api.types.is_numeric_dtype(col_data) else None,
                "sample_values": col_data.dropna().unique()[:5].tolist()
            }
        profile[sheet.title] = sheet_profile
    return profile

def scan_and_process_supabase(query_log=None):
    st.info("Scanning Supabase storage for supported files...")
    # Load global column aliases from Supabase
    global_aliases = load_global_aliases()
    updated_global_aliases = global_aliases.copy()

    files = list_supported_files()
    for file_info in files:
        file = file_info["name"]
        ext = file.split(".")[-1].lower()
        if ext not in ["docx", "xlsx", "pptx", "pdf"]:
            continue

        # Download file from Supabase
        file_stream = download_file(file)
        metadata = {}
        if ext == "docx":
            # Save to temp and process
            with open("temp.docx", "wb") as f:
                f.write(file_stream.read())
            metadata = extract_metadata_from_docx("temp.docx")
            raw_text = extract_text_for_metadata("temp.docx")
        elif ext == "xlsx":
            with open("temp.xlsx", "wb") as f:
                f.write(file_stream.read())
            metadata = extract_metadata_from_xlsx("temp.xlsx")
            raw_text = extract_text_for_metadata("temp.xlsx")
        elif ext == "pptx":
            with open("temp.pptx", "wb") as f:
                f.write(file_stream.read())
            metadata = extract_metadata_from_pptx("temp.pptx")
            raw_text = extract_text_for_metadata("temp.pptx")
        elif ext == "pdf":
            with open("temp.pdf", "wb") as f:
                f.write(file_stream.read())
            metadata = extract_metadata_from_pdf("temp.pdf")
            raw_text = extract_text_for_metadata("temp.pdf")
        else:
            continue

        gpt_metadata = auto_generate_metadata(raw_text[:3000] if raw_text else "")
        for key in ["title", "category", "tags", "summary"]:
            metadata[key] = gpt_metadata.get(key, metadata.get(key, ""))

        # Structural metadata
        structural_meta = extract_structural_metadata(raw_text, f".{ext}")
        metadata.update(structural_meta)

        profile_data = {}
        column_aliases = {}
        if ext == "xlsx":
            try:
                profile_data = profile_excel("temp.xlsx")
            except Exception as e:
                st.warning(f"Profile data error: {e}")
            column_aliases = map_columns_to_concepts(metadata["columns"], updated_global_aliases)
            updated_global_aliases.update(column_aliases)

        summary_text = metadata.get("summary", "")
        embedding = get_embedding(summary_text) if summary_text else None

        usage_stats = None
        # No local usage stats; optionally load from Supabase if needed

        metadata["last_indexed"] = datetime.now().isoformat()
        metadata["last_modified"] = datetime.now().isoformat()

        json_data = generate_json(metadata, file, profile_data, embedding, usage_stats, column_aliases)
        save_metadata(file, json_data)
        st.success(f"✅ Metadata saved to Supabase: {file}")

    update_global_aliases(updated_global_aliases)
def extract_metadata_from_pptx(pptx_path):
    prs = Presentation(pptx_path)
    metadata = {}
    if prs.slides:
        first_slide = prs.slides[0]
        for shape in first_slide.shapes:
            if shape.has_text_frame:
                for line in shape.text_frame.text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip().lower()] = value.strip()
    return metadata

def extract_metadata_from_pdf(pdf_path):
    metadata = {}
    with pdfplumber.open(pdf_path) as pdf:
        lines = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split('\n'))
            if len(lines) >= 10:
                break
        for line in lines[:10]:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip().lower()] = value.strip()
    return metadata

def extract_structural_metadata(raw_text, ext):
    if ext == ".docx":
        return {"section_headings": [line.strip() for line in raw_text.splitlines() if line.strip().endswith(":") or line.strip().istitle()]}
    elif ext == ".pdf":
        return {"section_headings": [line.strip() for line in raw_text.splitlines() if line.strip().endswith(":") or line.strip().istitle()]}
    elif ext == ".pptx":
        return {"slide_titles": [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 0]}
    return {}

def generate_json(metadata, original_filename, profile_data=None, embedding=None, usage_stats=None, column_aliases=None):
    return {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "date": metadata.get("date", str(datetime.today().date())),
        "category": metadata.get("category", ""),
        "tags": metadata.get("tags", []) if isinstance(metadata.get("tags"), list)
        else [tag.strip() for tag in metadata.get("tags", "").split(',') if tag.strip()] if "tags" in metadata else [],
        "summary": metadata.get("summary", ""),
        "source_file": original_filename,
        "profile_data": profile_data or {},
        "embedding": embedding.tolist() if embedding is not None else [],
        "usage_stats": usage_stats or {"times_used": 0, "last_used": None, "most_common_queries": []},
        "column_aliases": column_aliases or {}
    }


