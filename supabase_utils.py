from supabase_config import supabase
from datetime import datetime
import json
import io
import os
from uuid import uuid4

# ---------- STORAGE UTILS ----------

def list_supported_files():
    response = supabase.storage.from_('llm-files').list()
    return [f for f in response if f['name'].lower().endswith(('.pdf', '.docx', '.xlsx', '.pptx'))]

def upload_file(file_path: str, file_bytes: bytes):
    return supabase.storage.from_('llm-files').upload(file_path, file_bytes, {"content-type": _guess_mime(file_path)})

def download_file(file_path: str):
    response = supabase.storage.from_('llm-files').download(file_path)
    return io.BytesIO(response)

def get_file_last_modified(file_path: str):
    try:
        files = supabase.storage.from_('llm-files').list()
        for f in files:
            if f['name'] == os.path.basename(file_path):
                return f['updated_at']
    except Exception:
        return None

# ---------- METADATA UTILS ----------


def load_metadata(source_file: str):
    try:
        result = supabase.table("metadata").select("*").eq("source_file", source_file).single().execute()
        return result.data if result.data else {}
    except Exception:
        return {}


def save_metadata(source_file: str, data: dict):
    data['source_file'] = source_file
    # Use upsert for robust insert/update
    supabase.table("metadata").upsert(data, on_conflict=["source_file"]).execute()
    # Always return the row with id
    result = supabase.table("metadata").select("*").eq("source_file", source_file).single().execute()
    return result.data if result.data else None



# ---------- GLOBAL ALIASES ----------

def load_global_aliases():
    try:
        result = supabase.table("column_aliases").select("*", count='exact').execute()
        aliases = {row['original']: row['alias'] for row in result.data}
        return aliases
    except Exception:
        return {}

def update_global_aliases(new_aliases: dict):
    if not new_aliases:
        print("⚠️ No aliases to update — skipping Supabase insert.")
        return

    supabase.table("column_aliases").delete().neq("original", "").execute()
    
    rows = [{"original": k, "alias": v} for k, v in new_aliases.items()]
    
    if not rows:
        print("⚠️ No rows generated for insertion — skipping insert.")
        return

    print("✅ Inserting column_aliases:", rows)
    return supabase.table("column_aliases").insert(rows).execute()

# ---------- LEARNED ANSWERS ----------

def load_learned_answers():
    try:
        result = supabase.table("learned_answers").select("*").execute()
        return {row['question']: row for row in result.data}
    except Exception:
        return {}

def save_learned_answers(data: dict):
    # Upsert one-by-one
    for question, row in data.items():
        row['question'] = question
        supabase.table("learned_answers").upsert(row, on_conflict=['question']).execute()

# ---------- MIME TYPE GUESS ----------

def _guess_mime(filename):
    ext = filename.lower().split(".")[-1]
    return {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    }.get(ext, "application/octet-stream")

def insert_embedding_chunk(file_id, chunk_id, chunk_text, embedding, token_count):
    from supabase_config import supabase
    return supabase.table("embedding_index").insert({
        "file_id": file_id,
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "embedding": embedding,
        "token_count": token_count,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

