# scripts/bootstrap_assistant.py
import os, json, sys
from openai import OpenAI
import yaml

# Reuse your robust register script
from scripts.register_assistant import (
    load_yaml, to_instruction_text, create_or_update_assistant,
    ensure_vector_store, attach_vector_store, save_meta, CONFIG_PATH
)

def sync_dropbox_to_vector_store(client: OpenAI, vector_store_id: str, dbx_root: str):
    """
    Minimal stub so this runs in your hosted environment.
    Replace internals with your existing sync code if you have one.
    """
    import dropbox
    DBX_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
    if not DBX_TOKEN:
        raise RuntimeError("DROPBOX_ACCESS_TOKEN missing")
    dbx = dropbox.Dropbox(oauth2_access_token=DBX_TOKEN)

    # list files (xlsx,csv,pdf,docx,pptx)
    exts = {".xlsx",".xls",".csv",".pdf",".docx",".pptx"}
    collected = []
    res = dbx.files_list_folder(dbx_root, recursive=True)
    entries = res.entries
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        entries += res.entries
    for e in entries:
        if isinstance(e, dropbox.files.FileMetadata):
            ext = os.path.splitext(e.name)[1].lower()
            if ext in exts and e.size <= 100*1024*1024:
                collected.append(e.path_lower)

    print(f"Found {len(collected)} files under {dbx_root}")

    # upload/attach to vector store
    uploaded = 0
    for path in collected:
        _, resp = dbx.files_download(path)
        file_obj = client.files.create(file=(os.path.basename(path), resp.content), purpose="assistants")
        client.beta.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_obj.id)
        uploaded += 1
    print(f"Uploaded {uploaded} files to vector store {vector_store_id}")

def main():
    client = OpenAI()
    cfg = load_yaml(CONFIG_PATH)
    instructions_text = to_instruction_text(cfg)

    assistant = create_or_update_assistant(client, cfg, instructions_text)

    store_name = os.getenv("VECTOR_STORE_NAME") or cfg.get("assistant_profile",{}).get("vector_store_name") or "SCIE-Ethos-Store"
    vs_id = ensure_vector_store(client, store_name)
    attach_vector_store(client, assistant.id, vs_id)
    meta = save_meta(assistant, vector_store_id=vs_id)

    # Dropbox → File Search sync
    dbx_root = os.getenv("DROPBOX_ROOT", "/Project_Root")
    sync_dropbox_to_vector_store(client, vs_id, dbx_root)

    print("✅ Assistant ready, vector store attached, and Dropbox files synced.")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
