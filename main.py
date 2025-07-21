# --- main.py (Refactored + Learned Answers + Alias Editor) ---

import streamlit as st
from datetime import datetime
from difflib import SequenceMatcher
from dateutil.parser import isoparse
from utils.metadata import load_metadata, save_metadata, load_global_aliases, update_global_aliases, load_learned_answers, save_learned_answers
from gdrive_utils import list_all_supported_files, get_file_last_modified, download_file, get_drive_service
from llm_client import get_embedding, cosine_similarity, answer_question, verify_answer, generate_llm_metadata, mine_for_root_causes
from utils.text_utils import chunk_text, extract_text_for_metadata, extract_structural_metadata
from modeling import predictive_modeling_prebuilt, predictive_modeling_guided, predictive_modeling_inference, build_and_run_model
from utils.excel_qa import excel_qa
from structured_qa import structured_data_qa
from utils.column_mapping import map_columns_to_concepts

# --- Google Drive Configuration ---
PROJECT_ROOT_FOLDER_ID = "1t1CcZzwsjOPMNKKMkdJd6kXhixTreNuY"  # Your main project folder
METADATA_FOLDER_ID = "1l7ReGDGypnmcgUMAJ07Wf6Qwim_FGEMQ"  # Your _metadata folder (CORRECTED)

def find_similar_learned_answer(query, learned_answers, threshold=0.85):
    """Find similar previously learned answers."""
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    for past_question, entry in learned_answers.items():
        if similarity(query, past_question) >= threshold:
            return entry
    return None

# --- Session setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "query_log" not in st.session_state:
    st.session_state.query_log = []

st.header("Ask a question about your knowledge base")

# --- Streamlit UI for editing column aliases ---
if st.checkbox("🔧 Edit Column Alias Mappings"):
    current_aliases = load_global_aliases(METADATA_FOLDER_ID)
    
    # Create an editable interface for aliases
    st.write("**Current Column Aliases:**")
    
    # Convert to editable format
    if current_aliases:
        alias_df = st.data_editor(
            current_aliases, 
            num_rows="dynamic", 
            key="alias_editor",
            use_container_width=True
        )
        
        if st.button("💾 Save Updated Aliases"):
            update_global_aliases(alias_df, METADATA_FOLDER_ID)
            st.success("Aliases updated successfully.")
            st.rerun()
    else:
        st.info("No aliases found. They will be created automatically as files are processed.")

# --- Load aliases and files ---
global_aliases = load_global_aliases(METADATA_FOLDER_ID) or {}  # Ensure it's never None
updated_global_aliases = global_aliases.copy()
learned_answers = load_learned_answers(METADATA_FOLDER_ID) or {}  # Ensure it's never None

all_files = list_all_supported_files(PROJECT_ROOT_FOLDER_ID)
all_chunks = []
embedding_cache = {}

# --- File indexing ---
st.write(f"📁 **Found {len(all_files)} files in Google Drive**")
if len(all_files) == 0:
    st.warning("⚠️ No files found in Google Drive. Please check your connection and folder permissions.")
    st.stop()

progress_bar = st.progress(0, text="Indexing files and building metadata...")
for idx, file in enumerate(all_files):
    file_name = file["name"]
    file_id = file["id"]
    ext = file_name.lower().split(".")[-1] if "." in file_name else "unknown"
    
    file_last_modified = get_file_last_modified(file_id)
    meta = load_metadata(file_name, METADATA_FOLDER_ID) or {"source_file": file_name}
    needs_index = True

    # --- Efficient re-indexing: skip if file hasn't changed since last_indexed
    if meta.get("last_indexed") and file_last_modified:
        try:
            # file_last_modified is ISO string from Drive, meta["last_indexed"] is ISO string
            if isoparse(file_last_modified) <= isoparse(meta["last_indexed"]):
                needs_index = False
        except Exception:
            pass

    if needs_index:
        try:
            if ext == "xlsx":
                text, sheet_names, columns_by_sheet = extract_text_for_metadata(file_id, max_ocr_pages=5, download_file_func=download_file, get_drive_service_func=get_drive_service)
            else:
                text = extract_text_for_metadata(file_id, max_ocr_pages=5, download_file_func=download_file, get_drive_service_func=get_drive_service)
                sheet_names, columns_by_sheet = [], {}

            if isinstance(text, tuple):
                text = text[0]

            if text and text.strip():
                meta["file_type"] = ext
                meta["file_size"] = file.get("size", None)
                meta["last_modified"] = file_last_modified
                meta["last_indexed"] = datetime.now().isoformat()

                # Excel-specific
                if ext == "xlsx":
                    meta["sheet_names"] = sheet_names
                    meta["columns_by_sheet"] = columns_by_sheet
                    all_columns = []
                    for cols in columns_by_sheet.values():
                        all_columns.extend(cols)
                    meta["columns"] = list(set(all_columns))
                    column_aliases = map_columns_to_concepts(meta["columns"], updated_global_aliases)
                    meta["column_aliases"] = column_aliases
                    updated_global_aliases.update(column_aliases)

                # LLM-generated summary/tags/category for ALL files
                llm_meta = generate_llm_metadata(text, ext)
                meta.update(llm_meta)
                # Structural metadata for all files
                meta.update(extract_structural_metadata(text, ext))
                
                st.write(f"✅ Calling save_metadata for: {file_name}")  # 🔍 DEBUG LOG
                save_metadata(file_name, meta, METADATA_FOLDER_ID)

                for chunk in chunk_text(text):
                    all_chunks.append((meta, chunk))

                # --- Embedding (based on summary)
                if meta.get("summary", ""):
                    embedding_cache[file_name] = get_embedding(meta["summary"])

        except Exception as e:
            st.warning(f"⚠️ Failed to process {file_name}: {e}")
            continue
    else:
        # Use existing metadata and skip re-indexing
        if "summary" in meta:
            for chunk in chunk_text(meta["summary"]):
                all_chunks.append((meta, chunk))
        if "embedding" in meta and meta["embedding"]:
            embedding_cache[file_name] = meta["embedding"]

    progress_bar.progress((idx + 1) / len(all_files), text=f"Indexed {idx+1}/{len(all_files)} files")
progress_bar.empty()
update_global_aliases(updated_global_aliases, METADATA_FOLDER_ID)

# --- Question form ---
with st.form("question_form", clear_on_submit=False):
    user_query = st.text_input("Your question:", value=st.session_state.get("last_query", ""))
    submit = st.form_submit_button("Ask")

if submit and user_query.strip():
    # --- Check for file count queries ---
    query_lower = user_query.lower()
    file_count_keywords = [
        "how many files", "number of files", "file count", "count files",
        "how many documents", "number of documents", "document count", "count documents",
        "files do you have", "documents do you have", "files are there", "documents are there",
        "total files", "total documents", "files in drive", "documents in drive"
    ]
    if any(phrase in query_lower for phrase in file_count_keywords):
        st.info(f"🔍 Detected file count query: '{user_query}'")
        file_types = {}
        for file in all_files:
            file_name = file["name"]
            if "." in file_name:
                ext = file_name.lower().split(".")[-1]
            else:
                ext = "no extension"
            file_types[ext] = file_types.get(ext, 0) + 1
        
        # Debug: Show file names to understand the issue
        st.write("**Debug - File Details:**")
        for i, file in enumerate(all_files[:10]):  # Show first 10 files
            file_name = file['name']
            mime_type = file.get('mimeType', 'Unknown')
            folder_path = file.get('folder_path', 'Root')
            
            # Better file type detection based on file extension
            if file_name.lower().endswith('.xlsx'):
                file_type = "Excel"
            elif file_name.lower().endswith('.pdf'):
                file_type = "PDF"
            elif file_name.lower().endswith('.docx'):
                file_type = "Word"
            elif 'folder' in mime_type:
                file_type = "Folder"
            else:
                file_type = f"Other"
                
            st.write(f"{i+1}. **{file_name}** - {file_type} (in {folder_path})")
        
        if len(all_files) > 10:
            st.write(f"... and {len(all_files) - 10} more files")
        
        # Better file type counting
        file_types = {}
        for file in all_files:
            file_name = file["name"]
            
            if file_name.lower().endswith('.xlsx'):
                file_type = "Excel files"
            elif file_name.lower().endswith('.pdf'):
                file_type = "PDF files"
            elif file_name.lower().endswith('.docx'):
                file_type = "Word documents"
            elif file_name.lower().endswith('.pptx'):
                file_type = "PowerPoint files"
            else:
                file_type = "Other files"
                
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        # Generate the answer
        answer = f"I found **{len(all_files)} items** in your Google Drive knowledge base:\n\n"
        for file_type, count in sorted(file_types.items()):
            answer += f"- {count} {file_type}\n"
        
        st.markdown("**Answer:**")
        st.markdown(answer)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.stop()
    
    # --- Check if already learned ---
    if user_query in learned_answers:
        cached = learned_answers[user_query]
        st.success("✅ Reused cached answer")
        st.markdown("**Answer:**")
        st.markdown(cached.get("answer", ""))
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": cached.get("answer", "")})
        st.stop()

    st.info("Processing your question. This may take a moment...")
    query_embedding = get_embedding(user_query)

    scored_chunks = []
    for meta, chunk in all_chunks:
        meta_text = " ".join([
            str(meta.get("title", "")),
            str(meta.get("category", "")),
            " ".join(meta.get("tags", [])),
            str(meta.get("summary", "")),
            chunk
        ]).lower()
        keyword_score = sum(word in meta_text for word in user_query.lower().split())
        sem_score = cosine_similarity(query_embedding, embedding_cache.get(meta["source_file"]))
        hybrid_score = 0.5 * keyword_score + 0.5 * sem_score
        scored_chunks.append((hybrid_score, meta, chunk, keyword_score, sem_score))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = scored_chunks[:3]
    context = "\n\n".join([chunk for _, _, chunk, _, _ in top_chunks])

    answer = answer_question(user_query, context)
    verification = verify_answer(answer, context, user_query)
    root_cause = mine_for_root_causes(user_query, all_chunks, top_chunks)

    st.markdown(f"**Answer:**\n\n{answer}")
    st.markdown(f"**Verification:** {verification}")
    st.markdown("**Root Cause Analysis:**")
    st.write(root_cause)

    # --- Predictive Modeling Options ---
    st.markdown("### 📊 Predictive Modeling Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚙️ Run Prebuilt Model"):
            model_result = predictive_modeling_prebuilt(user_query, top_chunks)
            st.write(model_result)
    
    with col2:
        if st.button("🛠 Build and Train Model"):
            with st.spinner("Generating and training your model..."):
                model_result = build_and_run_model(user_query, top_chunks)
                st.write(model_result)
    
    with col3:
        if st.button("🧠 Simulate Prediction"):
            model_result = predictive_modeling_inference(user_query, top_chunks)
            st.write(model_result)

    # --- Advanced Analytics ---
    if st.button("📈 Run Structured Data Analysis"):
        structured_data_qa(user_query, top_chunks)

    # --- Cache answer ---
    learned_answers[user_query] = {
        "answer": answer,
        "files_used": [meta.get("source_file") for _, meta, _, _, _ in top_chunks],
        "scores": [
            {
                "file": meta.get("source_file"),
                "score": score,
                "keyword": kw,
                "semantic": sem
            }
            for score, meta, _, kw, sem in top_chunks
        ]
    }
    save_learned_answers(learned_answers, METADATA_FOLDER_ID)

    # --- Chart option ---
    top_meta = top_chunks[0][1] if top_chunks else None
    top_file = top_meta.get("source_file") if top_meta else None
    if top_file and top_file.lower().endswith('.xlsx'):
        st.markdown("---")
        st.markdown("**You can generate a chart from the top Excel file:**")
        if st.button("Show chart for top Excel file"):
            column_aliases = top_meta.get("column_aliases", {})
            excel_qa(top_file, user_query, column_aliases)

    # --- Log interaction ---
    st.session_state.last_query = user_query
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.query_log.append({
        "timestamp": datetime.now().isoformat(),
        "question": user_query,
        "files_used": [meta.get("source_file") for _, meta, _, _, _ in top_chunks],
        "scores": [
            {
                "file": meta.get("source_file"),
                "score": score,
                "keyword": kw,
                "semantic": sem
            }
            for score, meta, _, kw, sem in top_chunks
        ],
        "answer": answer,
        "verification": verification,
        "root_cause_analysis": root_cause,
    })

# --- Chat History and Query Log ---
with st.expander("Show Query Log"):
    for entry in st.session_state.query_log:
        st.write(entry)

with st.expander("Show Chat History"):
    for msg in st.session_state.chat_history:
        st.write(f"{msg['role'].capitalize()}: {msg['content']}")

