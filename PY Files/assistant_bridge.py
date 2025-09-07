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

def run_query(question: str, intent_hint: str | None = None, thread_id: str | None = None) -> dict:
    client = OpenAI()
    
    # Try to load assistant metadata from multiple sources
    assistant_id = None
    
    # 1. Try environment variable
    assistant_id = os.getenv("ASSISTANT_ID")
    
    # 2. Try local file
    if not assistant_id:
        try:
            with open("prompts/assistant.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
                assistant_id = meta["assistant_id"]
        except FileNotFoundError:
            pass
    
    # 3. Try Dropbox config
    if not assistant_id and _FILE_SYNC_AVAILABLE:
        try:
            from dbx_utils import dbx_read_json
            meta = dbx_read_json("/prompts/assistant.json")
            if meta and meta.get("assistant_id"):
                assistant_id = meta["assistant_id"]
        except Exception:
            pass
    
    if not assistant_id:
        raise RuntimeError("No assistant ID found. Run dropbox_sync.py first to create the assistant.")

    # Reuse existing thread or create new one
    if thread_id:
        thread = client.beta.threads.retrieve(thread_id)
    else:
        thread = client.beta.threads.create()
    
    # Create the user message
    message_content = question
    
    # If intent hint suggests data analysis, try to attach relevant Excel/CSV files
    if intent_hint and intent_hint in ["root_cause", "forecast", "movement_analysis", "eda"]:
        try:
            # Try to find relevant data files from Dropbox
            if _FILE_SYNC_AVAILABLE:
                from dbx_utils import list_data_files, read_file_bytes
                
                # Look for relevant data files
                data_folders = ["04_Data/01_Cleansed_Files", "04_Data/00_Raw_Files"]
                attached_files = []
                
                for folder in data_folders:
                    try:
                        files = list_data_files(folder)
                        # Attach first few relevant files
                        for file_info in files[:3]:  # Limit to 3 files
                            file_path = file_info["path_lower"]
                            file_content = read_file_bytes(file_path)
                            
                            # Upload file to OpenAI
                            uploaded_file = client.files.create(
                                file=file_content,
                                purpose="assistants"
                            )
                            attached_files.append(uploaded_file.id)
                            
                            print(f"📎 Attached file: {file_info['name']}")
                    except Exception as e:
                        print(f"⚠️ Could not attach files from {folder}: {e}")
                        continue
                
                # Create message with file attachments
                if attached_files:
                    client.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=message_content,
                        file_ids=attached_files
                    )
                else:
                    client.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=message_content
                    )
            else:
                # No file sync available, create message without attachments
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=message_content
                )
        except Exception as e:
            print(f"⚠️ File attachment failed: {e}")
            # Fallback to message without attachments
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message_content
            )
    else:
        # No data analysis intent, create message without attachments
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message_content
        )

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

def run_query_with_files(question: str, dropbox_paths: list[str], thread_id: str | None = None) -> dict:
    client = OpenAI()
    with open("prompts/assistant.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assistant_id = meta["assistant_id"]

    # Reuse existing thread or create new one
    if thread_id:
        thread = client.beta.threads.retrieve(thread_id)
    else:
        thread = client.beta.threads.create()

    # Use Code Interpreter instead of File Search for Excel files
    debug_info = {"file_sync_attempted": False, "files_uploaded": [], "code_interpreter_files": [], "sync_error": None}
    
    if _FILE_SYNC_AVAILABLE and dropbox_paths:
        debug_info["file_sync_attempted"] = True
        try:
            # Upload files directly for Code Interpreter (not File Search)
            from dbx_utils import upload_dropbox_file_to_openai
            file_ids = []
            for path in dropbox_paths:
                file_id = upload_dropbox_file_to_openai(path, purpose="assistants")
                if file_id:
                    file_ids.append(file_id)
            
            debug_info["files_uploaded"] = file_ids
            debug_info["code_interpreter_files"] = file_ids
            
            # Attach files to the thread for Code Interpreter
            if file_ids:
                client.beta.threads.update(
                    thread_id=thread.id,
                    tool_resources={"code_interpreter": {"file_ids": file_ids}}
                )
                debug_info["code_interpreter_attached"] = True
            else:
                debug_info["sync_error"] = "No files uploaded successfully"
                
        except Exception as e:
            debug_info["sync_error"] = str(e)
            print(f"File upload failed: {e}")

    # Enhanced prompt for Excel analysis
    excel_prompt = f"""
    {question}
    
    I have uploaded Excel files with supply chain/inventory data. Please analyze the data using Code Interpreter to:
    1. Load and examine the Excel file structure
    2. Identify WIP, inventory, and relevant financial columns
    3. Perform the requested analysis with specific numbers
    4. Show calculations and provide data-driven insights
    
    Use pandas to read the Excel file and provide precise numerical answers based on the actual data.
    """

    # Post the enhanced question
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=excel_prompt)

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


# NEW: Dual-answer function for Chat Assistant with KB + Data integration
def run_query_dual(
    messages: list[dict] | None = None,
    prompt: str | None = None, 
    thread_id: str | None = None,
    kb_scope_dbx: str | None = None,
    data_scope_dbx: str | None = None
) -> dict:
    """
    Enhanced query function that provides both document-based and AI-synthesized answers.
    
    Args:
        messages: Chat message history (not used yet, but reserved for future)
        prompt: Current user question
        thread_id: Existing thread ID to continue conversation
        kb_scope_dbx: Dropbox path to knowledge base (e.g., "/Project_Root/06_LLM_Knowledge_Base")
        data_scope_dbx: Dropbox path to data files (e.g., "/Project_Root/04_Data")
    
    Returns:
        Dict with dual answers, sources, confidence, and thread_id
    """
    client = OpenAI()
    
    # Use prompt if provided, otherwise extract from messages
    if prompt is None and messages:
        # Extract the latest user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if user_messages:
            prompt = user_messages[-1].get("content", "")
    
    if not prompt:
        return {
            "answer": "### AI Answer\nNo question provided.",
            "sources": {},
            "confidence": 0.0,
            "thread_id": thread_id or ""
        }
    
    try:
        # Load assistant configuration
        assistant_id = None
        try:
            with open("prompts/assistant.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
                assistant_id = meta["assistant_id"]
        except Exception:
            assistant_id = os.getenv("ASSISTANT_ID")
        
        if not assistant_id:
            # Fallback to basic query if no assistant configured
            return run_query(prompt, thread_id=thread_id)
        
        # Get or create thread
        if thread_id:
            thread = client.beta.threads.retrieve(thread_id)
        else:
            thread = client.beta.threads.create()
        
        # Get file candidates if KB/Data paths provided
        kb_files = []
        data_files = []
        
        if kb_scope_dbx or data_scope_dbx:
            try:
                from dropbox_kb_sync import list_kb_candidates
                
                candidates = list_kb_candidates(
                    kb_scope_dbx or "/Project_Root/06_LLM_Knowledge_Base",
                    data_scope_dbx or "/Project_Root/04_Data"
                )
                
                kb_files = candidates.get('kb_docs', [])
                data_files = candidates.get('data_files', [])
                
                
            except Exception as e:
                print(f"Warning: Could not get KB candidates: {e}")
        
        # Attach relevant files to assistant (simplified for now)
        file_ids = []
        if kb_files or data_files:
            # Try with dbx_utils if available
            if _FILE_SYNC_AVAILABLE:
                try:
                    from dbx_utils import upload_dropbox_file_to_openai
                    
                    # Upload a few relevant files
                    for file_info in (kb_files + data_files)[:5]:  # Limit to 5 files for performance
                        file_path = file_info['path']
                        try:
                            file_id = upload_dropbox_file_to_openai(file_path, purpose="assistants")
                            if file_id:
                                file_ids.append(file_id)
                        except Exception as e:
                            print(f"Failed to upload {file_info['name']}: {e}")
                            
                except Exception as e:
                    print(f"Warning: dbx_utils file upload failed: {e}")
            
            # Fallback: Create a summary of available documents for context
            else:
                # Group files by folder for better organization
                from dropbox_kb_sync import get_folder_structure
                folder_structure = get_folder_structure(kb_files)
                
                doc_summary = "Available Knowledge Base Documents:\n\n"
                for folder, docs in folder_structure.items():
                    doc_summary += f"**{folder}:**\n"
                    for doc in docs[:5]:  # Limit per folder
                        doc_summary += f"- {doc['name']} ({doc.get('size', 0)} bytes)\n"
                    doc_summary += "\n"
                
                if data_files:
                    doc_summary += "**Data Files:**\n"
                    for doc in data_files[:5]:
                        doc_summary += f"- {doc['name']} ({doc.get('size', 0)} bytes)\n"
                
                # Add document listing as context (fallback when files can't be uploaded)
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=f"CONTEXT: {doc_summary}\n\nNote: Document contents are not directly accessible for this query."
                )
        
        # DOCUMENT PASS: Try to answer from documents first
        doc_answer = "insufficient evidence in KB"
        doc_sources = {}
        doc_confidence = 0.0
        
        if file_ids:  # Only try document pass if we actually uploaded files
            try:
                # Add document-focused message
                doc_prompt = f"""Answer this question strictly from the attached documents and data files. 
                
Question: {prompt}

Requirements:
- Only use information directly found in the documents
- Always cite your sources with specific document names
- If evidence is insufficient, state "insufficient evidence in KB" and explain what's missing
- Provide at least 2 citations for a complete answer

Answer:"""
                
                client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user", 
                    content=doc_prompt
                )
                
                # Run with document focus
                run = client.beta.threads.runs.create(
                    thread_id=thread.id, 
                    assistant_id=assistant_id,
                    instructions="Focus on document content. Cite sources. Be strict about evidence requirements."
                )
                
                # Poll for completion
                terminal_states = {"completed", "failed", "cancelled", "expired"}
                while run.status not in terminal_states:
                    time.sleep(1)
                    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                
                doc_answer = _extract_answer(client, thread.id)
                
                # Simple confidence based on presence of citations
                citation_count = doc_answer.count("(") + doc_answer.count("[")  # Simple citation detection
                if "insufficient evidence" not in doc_answer.lower() and citation_count >= 2:
                    doc_confidence = 0.8
                else:
                    doc_confidence = 0.3
                    
            except Exception as e:
                print(f"Document pass failed: {e}")
                doc_answer = "insufficient evidence in KB - document search error"
        
        elif kb_files or data_files:
            # We have document metadata but no file access
            doc_answer = f"insufficient evidence in KB - Found {len(kb_files)} KB documents and {len(data_files)} data files, but cannot access content. Please ensure Dropbox integration is properly configured."
        
        # AI PASS: Get synthesized answer
        ai_answer = "Unable to provide AI analysis."
        ai_confidence = 0.0
        
        try:
            # Add AI-focused message
            ai_prompt = f"""Provide a comprehensive analysis of this question using your knowledge and reasoning:

Question: {prompt}

Provide a thoughtful, detailed response that synthesizes information and provides insights."""
            
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=ai_prompt
            )
            
            # Run for AI synthesis
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            
            # Poll for completion
            while run.status not in terminal_states:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            
            ai_answer = _extract_answer(client, thread.id)
            ai_confidence = 0.7  # Default AI confidence
            
        except Exception as e:
            print(f"AI pass failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Compose dual response
        final_answer = f"""### Document Answer
{doc_answer}

### AI Answer
{ai_answer}"""
        
        # Combine sources (simplified for now)
        combined_sources = {
            "kb_sources": [{"name": f["name"], "path": f["path"]} for f in kb_files[:5]],
            "data_sources": [{"name": f["name"], "path": f["path"]} for f in data_files[:3]],
            "file_sources": []
        }
        
        # Use higher confidence score
        final_confidence = max(doc_confidence, ai_confidence)
        
        return {
            "answer": final_answer,
            "sources": combined_sources,
            "confidence": final_confidence,
            "thread_id": thread.id,
            "doc_confidence": doc_confidence,
            "ai_confidence": ai_confidence
        }
        
    except Exception as e:
        print(f"Dual query failed, falling back to basic query: {e}")
        import traceback
        traceback.print_exc()
        
        # Graceful fallback to existing function
        try:
            basic_response = run_query(prompt, thread_id=thread_id)
            # Convert to dual format for consistency
            return {
                "answer": f"""### Document Answer
insufficient evidence in KB - fallback mode

### AI Answer  
{basic_response.get('answer', 'Unable to provide analysis.')}""",
                "sources": basic_response.get('sources', {}),
                "confidence": basic_response.get('confidence', 0.5),
                "thread_id": basic_response.get('thread_id', thread_id or ''),
                "doc_confidence": 0.0,
                "ai_confidence": basic_response.get('confidence', 0.5)
            }
        except Exception as fallback_error:
            print(f"Even basic fallback failed: {fallback_error}")
            traceback.print_exc()
            return {
                "answer": f"""### Document Answer
insufficient evidence in KB

### AI Answer
I apologize, but I encountered an error processing your question. Please try again.""",
                "sources": {},
                "confidence": 0.0,
                "thread_id": thread_id or "",
                "doc_confidence": 0.0,
                "ai_confidence": 0.0
            }


# ── Router Bridge Function ──────────────────────────────────────────────────
from typing import Any, Dict
import os
from router_client import RouterClient
from orchestrator import route_from_router

CONF_MIN = float(os.getenv("ROUTER_CONF_MIN", "0.55"))

def run_via_router(user_text: str, session_state: Any = None) -> Dict[str, Any]:
    """
    Classify with Router → hand off to orchestrator.route_from_router().
    Zero UI changes elsewhere; call this only when feature flag is on.
    """
    # If we asked a clarifier last turn, merge with original (Step 8)
    if session_state is not None and session_state.get("_router_pending"):
        base = session_state["_router_pending"].get("original_text", "")
        user_text = f"{base}\nClarification: {user_text}".strip()

    # SAFETY: try Router; on failure, run DP directly with original text
    try:
        contract = RouterClient().classify(user_text)
        conf = float(contract.get("confidence", 0.0) or 0.0)
    except Exception as e:
        # Record error for telemetry; then fall back to DP
        if session_state is not None:
            session_state["_router_error"] = str(e)[:200]
        from dp_orchestrator import DataProcessingOrchestrator
        dp_orchestrator = DataProcessingOrchestrator()
        res = dp_orchestrator.process_dp_query(user_text, session_state)
        # flag the fallback for logs
        res.setdefault("_router_meta", {})["fallback"] = True
        res["_router_meta"]["original_text"] = user_text
        return res

    # Low-confidence clarifier (Step 6)
    if conf < CONF_MIN:
        missing = contract.get("missing") or []
        intent  = (contract.get("intent") or "unknown").lower()
        hints   = contract.get("files_hint") or []
        if session_state is not None:
            session_state["_router_pending"] = {"original_text": user_text, "contract": contract}

        lines = []
        if missing: 
            lines.append("• " + "\n• ".join(str(m) for m in missing[:3]))
        else:
            lines.append(
                "Which SKU/plant/period should I forecast?" if intent=="forecast"
                else "Which file(s) or period (e.g., R401 Q2-2025) should I analyze?" if intent=="rca"
                else "Which document(s) should I read?"
            )
        if hints: 
            lines.append(f"(I'll prioritize: {', '.join(hints[:3])})")

        return {"answer": "I can run this, but need a quick confirmation:\n" + "\n".join(lines),
                "intent": "clarifier", "confidence": conf,
                "_router_meta": {"contract": contract, "confidence": conf}}

    # Confidence OK → clear pending and run DP
    if session_state is not None and session_state.get("_router_pending"):
        session_state.pop("_router_pending", None)

    res = route_from_router(contract, session_state)
    # attach Router meta for UI logger
    res.setdefault("_router_meta", {})
    res["_router_meta"]["contract"] = contract
    res["_router_meta"]["confidence"] = conf
    
    # Learn from successful DP runs (feature-flagged)
    try:
        if os.getenv("MEMORY_ENABLED","0") in ("1","true","True"):
            from memory_store import learn_router_success
            learn_router_success(contract, res, conf_min=float(os.getenv("MEMORY_CONF_MIN","0.70")))
    except Exception:
        pass  # learning must never affect responses
    
    return res