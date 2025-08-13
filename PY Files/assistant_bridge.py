# PY Files/assistant_bridge.py
import os, json, time
from openai import OpenAI
from confidence import score_ravc, should_abstain  # moved from app.confidence :contentReference[oaicite:2]{index=2}

# NEW: Cloud-only file sync helpers
try:
    from dbx_utils import (
        prepare_file_search_from_dropbox,
        attach_vector_store_to_thread,
    )
    _FILE_SYNC_AVAILABLE = True
except Exception:
    _FILE_SYNC_AVAILABLE = False

def auto_model(intent: str) -> str:
    if intent in {"root_cause", "forecasting", "scenario", "anomaly", "exec_summary"}:
        return "gpt-4o"
    return "gpt-4o-mini"

def _extract_answer(client: OpenAI, thread_id: str) -> str:
    """
    Robustly extract the latest assistant message text.
    Handles multiple content parts and non-text parts gracefully.
    """
    msgs = client.beta.threads.messages.list(thread_id=thread_id, order="asc").data
    assistants = [m for m in msgs if getattr(m, "role", "") == "assistant"]
    if not assistants:
        return ""
    last = assistants[-1]
    parts = getattr(last, "content", []) or []
    # prefer all text parts concatenated
    text_chunks = []
    for p in parts:
        try:
            if getattr(p, "type", "") == "text" and hasattr(p, "text") and getattr(p.text, "value", None):
                text_chunks.append(p.text.value)
        except Exception:
            continue
    if text_chunks:
        return "\n\n".join(text_chunks)
    # fallback to first text-ish thing if structure changes
    try:
        return parts[0].text.value  # original approach :contentReference[oaicite:3]{index=3}
    except Exception:
        return ""

def run_query(question: str, intent_hint: str | None = None) -> dict:
    client = OpenAI()
    with open("config/assistant.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assistant_id = meta["assistant_id"]

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=question)

    # Let the Assistant use File Search + Code Interpreter
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    # Poll until complete (hosted runner does this synchronously)
    terminal_states = {"completed", "failed", "cancelled", "expired"}
    while True:
        r = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if r.status in terminal_states:
            break
        time.sleep(1.2)

    # Extract answer text defensively
    answer = _extract_answer(client, thread.id)

    # Simple confidence example; replace with real signals
    conf = score_ravc(recency=0.8, alignment=0.8, variance=0.2, coverage=0.8)  # :contentReference[oaicite:4]{index=4}
    if should_abstain(conf["score"], threshold=0.52):  # :contentReference[oaicite:5]{index=5}
        answer = "I’m not confident enough to answer. Please refine the question."

    # Log to S3 (best-effort; never block the reply)
    bucket = os.getenv("S3_BUCKET")
    prefix = os.getenv("S3_PREFIX", "project-root")
    if bucket:
        try:
            import boto3, datetime as dt
            s3 = boto3.client("s3")
            log = {
                "ts": dt.datetime.utcnow().isoformat() + "Z",
                "q": question,
                "answer": answer,
                "confidence": conf,
            }
            key = f"{prefix}/logs/query_log.jsonl"
            # append-ish write (racy but fine for low concurrency)
            try:
                old = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            except Exception:
                old = ""
            s3.put_object(Bucket=bucket, Key=key, Body=(old + json.dumps(log) + "\n").encode("utf-8"))
        except Exception:
            # swallow logging errors
            pass

    return {"answer": answer, "confidence": conf}

# NEW: Variant that syncs selected Dropbox cleansed files to OpenAI and attaches a vector store

def run_query_with_files(question: str, dropbox_paths: list[str]) -> dict:
    client = OpenAI()
    with open("config/assistant.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assistant_id = meta["assistant_id"]

    thread = client.beta.threads.create()

    # If file sync helpers are available, attach a vector store at the thread level
    debug_info = {"file_sync_attempted": False, "files_uploaded": [], "vector_store_created": False, "sync_error": None}
    
    if _FILE_SYNC_AVAILABLE and dropbox_paths:
        debug_info["file_sync_attempted"] = True
        try:
            fs = prepare_file_search_from_dropbox(dropbox_paths, vs_name="Ethos_FileStore")
            debug_info["files_uploaded"] = fs.get("file_ids", [])
            debug_info["vector_store_created"] = bool(fs.get("vector_store_id"))
            if fs.get("vector_store_id"):
                attach_result = attach_vector_store_to_thread(thread.id, fs["vector_store_id"])  # thread-scoped
                debug_info["vector_store_attached"] = attach_result
            else:
                debug_info["sync_error"] = "No vector store created"
        except Exception as e:
            # Proceed without file search if sync fails
            debug_info["sync_error"] = str(e)
            print(f"File-search sync skipped: {e}")

    # Post the question
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=question)

    # Run with the configured Assistant (has code_interpreter + file_search enabled)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    # Poll until complete
    terminal_states = {"completed", "failed", "cancelled", "expired"}
    while True:
        r = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if r.status in terminal_states:
            break
        time.sleep(1.2)

    answer = _extract_answer(client, thread.id)
    conf = score_ravc(recency=0.8, alignment=0.8, variance=0.2, coverage=0.8)
    if should_abstain(conf["score"], threshold=0.52):
        answer = "I’m not confident enough to answer. Please refine the question."

    return {"answer": answer, "confidence": conf, "thread_id": thread.id, "debug_file_sync": debug_info}

