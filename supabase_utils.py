from supabase_config import supabase
from datetime import datetime

def insert_metadata(metadata: dict):
    response = supabase.table("metadata").insert(metadata).execute()
    return response

def insert_column_aliases(file_id: str, aliases: list[dict]):
    rows = [{"file_id": file_id, **alias} for alias in aliases]
    return supabase.table("column_aliases").insert(rows).execute()

def insert_embedding_chunk(file_id: str, chunk_id: str, chunk_text: str, embedding: list[float], token_count: int):
    return supabase.table("embedding_index").insert({
        "file_id": file_id,
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "embedding": embedding,
        "token_count": token_count,
        "created_at": datetime.utcnow().isoformat()
    }).execute()
