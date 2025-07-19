"""
LLM Metadata Sidecar Generator Script (Extended)
-------------------------------------------------
This script scans a project directory recursively for supported document types,
extracts structured metadata from the document itself (if present),
or falls back to OpenAI GPT to generate metadata.
It then saves a sidecar .json file into a `_metadata/` subfolder next to each file.

Supported: .docx, .xlsx, .pptx, .pdf
"""

import os
import json
from datetime import datetime
from docx import Document
import openpyxl
from pptx import Presentation
import pdfplumber
from openai import OpenAI
import numpy as np
import pandas as pd
from gdrive_utils import (
    list_all_supported_files, download_file, upload_json_file
)

client = OpenAI(api_key="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")  # Replace with your real key

GLOBAL_ALIAS_PATH = "./Project_Root/01_Project_Plan/global_column_aliases.json"
PROJECT_ROOT_ID = "1t1CcZzwsjOPMNKKMkdJd6kXhixTreNuY"  # Your Project_Root folder ID

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

def extract_text_for_metadata(path):
    if path.endswith(".docx"):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif path.endswith(".xlsx"):
        wb = openpyxl.load_workbook(path)
        text = []
        for sheet in wb.worksheets:
            text.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(max_row=5):
                row_text = [str(cell.value) for cell in row if cell.value]
                if row_text:
                    text.append(" | ".join(row_text))
        return "\n".join(text)
    elif path.endswith(".pptx"):
        prs = Presentation(path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text.append(shape.text)
        return "\n".join(text)
    elif path.endswith(".pdf"):
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    return ""

def auto_generate_metadata(prompt_text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a metadata assistant. Generate structured metadata for internal business documents.\n"
                    "Return only valid JSON with the following fields:\n"
                    "- title (short)\n"
                    "- category (1-2 words)\n"
                    "- tags (array of 3–7 keywords)\n"
                    "- summary (1 paragraph overview of the content; if the document is a spreadsheet, summarize what the data represents, including sheet and column names if possible)\n\n"
                    "Do not include markdown formatting, explanations, or anything other than valid JSON."
                )
            },
            {
                "role": "user",
                "content": f"Document content:\n{prompt_text[:3000]}\n\nReturn metadata JSON only."
            }
        ],
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```json") or raw.startswith("```"):
        raw = raw.strip("`").replace("json", "").strip()

    try:
        metadata = json.loads(raw)
        metadata["title"] = metadata.get("title", "").strip()
        metadata["category"] = metadata.get("category", "").strip()
        metadata["tags"] = metadata.get("tags", []) if isinstance(metadata.get("tags", []), list) else []
        metadata["summary"] = metadata.get("summary", "").strip()
        return metadata
    except Exception as e:
        print("⚠️ GPT returned invalid JSON:", raw)
        raise e

def extract_metadata_from_docx(doc_path):
    doc = Document(doc_path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    metadata = {}
    for line in lines[:10]:
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip().lower()] = value.strip()
    return metadata

def extract_metadata_from_xlsx(file_path):
    wb = openpyxl.load_workbook(file_path, read_only=True)
    metadata = {}
    metadata["sheet_names"] = wb.sheetnames
    columns_by_sheet = {}
    for sheet in wb.worksheets:
        headers = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
        columns_by_sheet[sheet.title] = [str(h) for h in headers if h]
    metadata["columns_by_sheet"] = columns_by_sheet
    all_columns = []
    for cols in columns_by_sheet.values():
        all_columns.extend(cols)
    metadata["columns"] = list(set(all_columns))
    ws = wb.active
    for row in ws.iter_rows(min_row=1, max_row=5, min_col=1, max_col=2):
        key = row[0].value
        value = row[1].value
        if key:
            metadata[str(key).strip().lower()] = str(value).strip() if value else ""
    return metadata

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

def get_gdrive_id_by_name(name, parent_id, is_folder=False):
    files = list_all_supported_files(parent_id)
    for f in files:
        if f["name"] == name:
            if is_folder and f["mimeType"] == "application/vnd.google-apps.folder":
                return f["id"]
            if not is_folder and f["mimeType"] != "application/vnd.google-apps.folder":
                return f["id"]
    return None

METADATA_FOLDER_ID = get_gdrive_id_by_name("_metadata", PROJECT_ROOT_ID, is_folder=True)

def scan_and_process(base_dir, query_log=None):
    print(f"Scanning {base_dir}...")

    # --- Load global column aliases ---
    if os.path.exists(GLOBAL_ALIAS_PATH):
        with open(GLOBAL_ALIAS_PATH, "r") as f:
            global_aliases = json.load(f)
    else:
        global_aliases = {}

    updated_global_aliases = global_aliases.copy()

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            full_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            base_name = os.path.splitext(file)[0]
            meta_dir = os.path.join(root, "_metadata")
            json_file_path = os.path.join(meta_dir, f"{base_name}.json")

            # --- Efficient re-indexing ---
            file_last_modified = os.path.getmtime(full_path)
            needs_index = True
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, "r") as f:
                        old = json.load(f)
                        last_indexed = old.get("last_indexed")
                        if last_indexed:
                            last_indexed_dt = datetime.fromisoformat(last_indexed)
                            if file_last_modified <= last_indexed_dt.timestamp():
                                needs_index = False
                except Exception:
                    pass

            if not needs_index:
                print(f"⏩ Skipped (not modified): {file}")
                continue

            metadata = {}

            if ext == ".docx":
                metadata = extract_metadata_from_docx(full_path)
            elif ext == ".xlsx":
                metadata = extract_metadata_from_xlsx(full_path)
            elif ext == ".pptx":
                metadata = extract_metadata_from_pptx(full_path)
            elif ext == ".pdf":
                metadata = extract_metadata_from_pdf(full_path)
            else:
                continue

            raw_text = extract_text_for_metadata(full_path)
            gpt_metadata = auto_generate_metadata(raw_text[:3000] if raw_text else "")

            # Always use LLM-generated summary/tags/category for robustness
            for key in ["title", "category", "tags", "summary"]:
                metadata[key] = gpt_metadata.get(key, metadata.get(key, ""))

            # --- Structural metadata ---
            structural_meta = extract_structural_metadata(raw_text, ext)
            metadata.update(structural_meta)

            profile_data = {}
            column_aliases = {}
            if ext == ".xlsx":
                try:
                    profile_data = profile_excel(full_path)
                except Exception as e:
                    print(f"Profile data error: {e}")
                # --- Column alias mapping ---
                column_aliases = map_columns_to_concepts(metadata["columns"], updated_global_aliases)
                updated_global_aliases.update(column_aliases)

            summary_text = metadata.get("summary", "")
            embedding = get_embedding(summary_text) if summary_text else None

            usage_stats = None
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, "r") as f:
                        old = json.load(f)
                        usage_stats = old.get("usage_stats", None)
                        if query_log and "tags" in metadata:
                            metadata["tags"] = enrich_tags(metadata["tags"], usage_stats or {})
                except Exception:
                    pass
            else:
                if query_log and "tags" in metadata:
                    metadata["tags"] = enrich_tags(metadata["tags"], query_log)

            # --- Add last_indexed and last_modified ---
            metadata["last_indexed"] = datetime.now().isoformat()
            metadata["last_modified"] = datetime.fromtimestamp(file_last_modified).isoformat()

            json_data = generate_json(metadata, file, profile_data, embedding, usage_stats, column_aliases)
            os.makedirs(meta_dir, exist_ok=True)
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            print(f"✅ Metadata saved: {json_file_path}")

    # --- Save updated global column aliases ---
    with open(GLOBAL_ALIAS_PATH, "w") as f:
        json.dump(updated_global_aliases, f, indent=2)

def scan_and_process_gdrive(project_root_id, query_log=None):
    print(f"Scanning Google Drive folder {project_root_id}...")

    # --- Load global column aliases ---
    alias_file_id = get_gdrive_id_by_name("global_column_aliases.json", METADATA_FOLDER_ID)
    if alias_file_id:
        alias_stream = download_file(alias_file_id)
        global_aliases = json.load(alias_stream)
    else:
        global_aliases = {}

    updated_global_aliases = global_aliases.copy()

    files = list_all_supported_files(project_root_id)
    for f in files:
        file_id = f["id"]
        file_name = f["name"]
        ext = os.path.splitext(file_name)[1].lower()
        base_name = os.path.splitext(file_name)[0]

        # Only process supported types
        if ext not in [".docx", ".xlsx", ".pptx", ".pdf"]:
            continue

        # Download file to memory
        file_stream = download_file(file_id)
        temp_path = f"/tmp/{file_name}"
        with open(temp_path, "wb") as tmpf:
            tmpf.write(file_stream.read())

        # --- Efficient re-indexing ---
        # Check for existing metadata in Drive
        json_file_name = f"{base_name}.json"
        meta_file_id = get_gdrive_id_by_name(json_file_name, METADATA_FOLDER_ID)
        needs_index = True
        if meta_file_id:
            meta_stream = download_file(meta_file_id)
            old = json.load(meta_stream)
            last_indexed = old.get("last_indexed")
            file_last_modified = f.get("modifiedTime")
            if last_indexed and file_last_modified:
                try:
                    from dateutil.parser import isoparse
                    if isoparse(file_last_modified) <= isoparse(last_indexed):
                        needs_index = False
                except Exception:
                    pass

        if not needs_index:
            print(f"⏩ Skipped (not modified): {file_name}")
            continue

        metadata = {}
        if ext == ".docx":
            metadata = extract_metadata_from_docx(temp_path)
        elif ext == ".xlsx":
            metadata = extract_metadata_from_xlsx(temp_path)
        elif ext == ".pptx":
            metadata = extract_metadata_from_pptx(temp_path)
        elif ext == ".pdf":
            metadata = extract_metadata_from_pdf(temp_path)

        raw_text = extract_text_for_metadata(temp_path)
        gpt_metadata = auto_generate_metadata(raw_text[:3000] if raw_text else "")

        for key in ["title", "category", "tags", "summary"]:
            metadata[key] = gpt_metadata.get(key, metadata.get(key, ""))

        structural_meta = extract_structural_metadata(raw_text, ext)
        metadata.update(structural_meta)

        profile_data = {}
        column_aliases = {}
        if ext == ".xlsx":
            try:
                profile_data = profile_excel(temp_path)
            except Exception as e:
                print(f"Profile data error: {e}")
            column_aliases = map_columns_to_concepts(metadata["columns"], updated_global_aliases)
            updated_global_aliases.update(column_aliases)

        summary_text = metadata.get("summary", "")
        embedding = get_embedding(summary_text) if summary_text else None

        usage_stats = None
        if meta_file_id:
            meta_stream = download_file(meta_file_id)
            old = json.load(meta_stream)
            usage_stats = old.get("usage_stats", None)
            if query_log and "tags" in metadata:
                metadata["tags"] = enrich_tags(metadata["tags"], usage_stats or {})
        else:
            if query_log and "tags" in metadata:
                metadata["tags"] = enrich_tags(metadata["tags"], query_log)

        from datetime import datetime
        metadata["last_indexed"] = datetime.now().isoformat()
        metadata["last_modified"] = f.get("modifiedTime")

        json_data = generate_json(metadata, file_name, profile_data, embedding, usage_stats, column_aliases)
        upload_json_file(METADATA_FOLDER_ID, json_file_name, json_data)
        print(f"✅ Metadata saved: {json_file_name}")

    # --- Save updated global column aliases ---
    upload_json_file(METADATA_FOLDER_ID, "global_column_aliases.json", updated_global_aliases)

if __name__ == "__main__":
    scan_and_process_gdrive(PROJECT_ROOT_ID)