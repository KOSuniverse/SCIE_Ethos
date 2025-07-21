import streamlit as st
from datetime import datetime
from difflib import SequenceMatcher
from dateutil.parser import isoparse

# --- Imports
from utils.metadata import (
    load_metadata, save_metadata, load_global_aliases,
    update_global_aliases, load_learned_answers, save_learned_answers
)
from utils.text_utils import chunk_text, extract_text_for_metadata, extract_structural_metadata
from utils.column_mapping import map_columns_to_concepts
from utils.gdrive import (
    list_all_supported_files, get_file_last_modified, download_file,
    get_metadata_folder_id, get_drive_service
)
from llm_client import get_embedding, cosine_similarity, answer_question, verify_answer, generate_llm_metadata, mine_for_root_causes
from modeling import predictive_modeling_prebuilt, predictive_modeling_inference, build_and_run_model
from structured_qa import structured_data_qa
from utils.excel_qa import excel_qa

# --- Session Setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "query_log" not in st.session_state:
    st.session_state.query_log = []

st.header("Ask a question about your knowledge base")

# --- Get Drive metadata folder
metadata_folder_id = get_metadata_folder_id()
if not metadata_folder_id:
    st.error("❌ Cannot continue without access to _metadata folder.")
    st.stop()

# --- Load aliases and answers
global_aliases = load_global_aliases(metadata_folder_id) or {}
updated_global_aliases = global_aliases.copy()
learned_answers = load_learned_answers(metadata_folder_id) or {}

# --- File loading
all_files = list_all_supported_files()
if not all_files:
    st.warning("⚠️ No files found in Google Drive.")
    st.stop()
st.success(f"📁 Found {len(all_files)} files")

# --- File indexing
all_chunks = []
embedding_cache = {}
progress_bar = st.progress(0, text="Indexing files...")

for idx, file in enumerate(all_files):
    file_name = file["name"]
    file_id = file["id"]
    ext = file_name.lower().split(".")[-1] if "." in file_name else "unknown"

    # ✅ Validate file is accessible (skip 404s)
    try:
        file_last_modified = get_file_last_modified(file_id)
    except Exception as e:
        st.warning(f"⚠️ Skipping file: {file_name} — reason: {e}")
        continue

    meta = load_metadata(file_name, metadata_folder_id) or {"source_file": file_name}
    needs_index = True

    if meta.get("last_indexed") and file_last_modified:
        try:
            if isoparse(file_last_modified) <= isoparse(meta["last_indexed"]):
                needs_index = False
        except Exception:
            pass

        if needs_index:
            if ext == "xlsx":
                text, sheet_names, columns_by_sheet = extract_text_for_metadata(
                    file_id, max_ocr_pages=5,
                    download_file_func=download_file,
                    get_drive_service_func=get_drive_service
                )
            else:
                text, sheet_names, columns_by_sheet = extract_text_for_metadata(
                    file_id, max_ocr_pages=5,
                    download_file_func=download_file,
                    get_drive_service_func=get_drive_service
                )

            if isinstance(text, tuple):
                text = text[0]

            if text and text.strip():
                meta.update({
                    "file_type": ext,
                    "file_size": file.get("size", None),
                    "last_modified": file_last_modified,
                    "last_indexed": datetime.now().isoformat()
                })

                if ext == "xlsx":
                    meta["sheet_names"] = sheet_names
                    meta["columns_by_sheet"] = columns_by_sheet
                    all_columns = [col for cols in columns_by_sheet.values() for col in cols]
                    meta["columns"] = list(set(all_columns))
                    column_aliases = map_columns_to_concepts(meta["columns"], updated_global_aliases)
                    meta["column_aliases"] = column_aliases
                    updated_global_aliases.update(column_aliases)

                # Generate LLM metadata
                llm_meta = generate_llm_metadata(text, ext)
                meta.update(llm_meta)
                meta.update(extract_structural_metadata(text, ext))

                save_metadata(file_name, meta, metadata_folder_id)

                for chunk, _, _ in chunk_text(text):
                    all_chunks.append((meta, chunk))

                if meta.get("summary", ""):
                    embedding_cache[file_name] = get_embedding(meta["summary"])

        else:
            # Use existing metadata
            if "summary" in meta:
                for chunk, _, _ in chunk_text(meta["summary"]):
                    all_chunks.append((meta, chunk))
            if "embedding" in meta:
                embedding_cache[file_name] = meta["embedding"]

    except Exception as e:
        st.warning(f"⚠️ Failed to process {file_name}: {e}")
        continue

    progress_bar.progress((idx + 1) / len(all_files), text=f"Indexed {idx+1}/{len(all_files)}")

progress_bar.empty()
update_global_aliases(updated_global_aliases, metadata_folder_id)

# --- Ask a question
with st.form("question_form", clear_on_submit=False):
    user_query = st.text_input("Your question:", value=st.session_state.get("last_query", ""))
    submit = st.form_submit_button("Ask")

if submit and user_query.strip():
    if user_query in learned_answers:
        cached = learned_answers[user_query]
        st.success("✅ Reused cached answer")
        st.markdown("**Answer:**")
        st.markdown(cached.get("answer", ""))
        st.stop()

    st.info("Processing your question...")
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

    # Modeling buttons
    if st.button("📊 Run Structured Data Analysis"):
        structured_data_qa(user_query, top_chunks)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⚙️ Run Prebuilt Model"):
            st.write(predictive_modeling_prebuilt(user_query, top_chunks))
    with col2:
        if st.button("🛠 Build and Train Model"):
            with st.spinner("Generating and training your model..."):
                st.write(build_and_run_model(user_query, top_chunks))
    with col3:
        if st.button("🧠 Simulate Prediction"):
            st.write(predictive_modeling_inference(user_query, top_chunks))

    # Save learned
    learned_answers[user_query] = {
        "answer": answer,
        "files_used": [meta.get("source_file") for _, meta, _, _, _ in top_chunks],
        "scores": [
            {"file": meta.get("source_file"), "score": score, "keyword": kw, "semantic": sem}
            for score, meta, _, kw, sem in top_chunks
        ]
    }
    save_learned_answers(learned_answers, metadata_folder_id)

