# --- main.py (Refactored + Learned Answers + Alias Editor) ---

import streamlit as st
from datetime import datetime
from difflib import SequenceMatcher
from utils.metadata import load_metadata, save_metadata, load_global_aliases, update_global_aliases, load_learned_answers, save_learned_answers
from utils.gdrive import list_all_supported_files, get_file_last_modified, download_file
from llm_client import get_embedding, cosine_similarity, answer_question, verify_answer, generate_llm_metadata, mine_for_root_causes
from utils.text_utils import chunk_text, extract_text_for_metadata, extract_structural_metadata
from modeling import predictive_modeling_prebuilt, predictive_modeling_guided, predictive_modeling_inference, build_and_run_model
from utils.excel_qa import excel_qa
from structured_qa import structured_data_qa
from utils.column_mapping import map_columns_to_concepts

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
    current_aliases = load_global_aliases()
    
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
            update_global_aliases(alias_df)
            st.success("Aliases updated successfully.")
            st.rerun()
    else:
        st.info("No aliases found. They will be created automatically as files are processed.")

# --- Load aliases and files ---
global_aliases = load_global_aliases()
updated_global_aliases = global_aliases.copy()
learned_answers = load_learned_answers()

all_files = list_all_supported_files()
all_chunks = []
embedding_cache = {}

# --- File indexing ---
progress_bar = st.progress(0, text="Indexing files and building metadata...")
for idx, file in enumerate(all_files):
    file_name = file["name"]
    file_id = file["id"]
    ext = file_name.lower().split(".")[-1]
    meta = load_metadata(file_name) or {"source_file": file_name}
    file_last_modified = get_file_last_modified(file_id)
    needs_index = True

    if meta.get("last_indexed") and file_last_modified:
        try:
            if file_last_modified <= datetime.fromisoformat(meta["last_indexed"]).timestamp():
                needs_index = False
        except Exception:
            pass

    if needs_index:
        try:
            text, sheet_names, columns_by_sheet = extract_text_for_metadata(file_id) if ext == "xlsx" else (extract_text_for_metadata(file_id), [], {})
            text = text[0] if isinstance(text, tuple) else text

            if text.strip():
                meta.update({
                    "file_type": ext,
                    "last_modified": datetime.fromtimestamp(file_last_modified).isoformat() if file_last_modified else None,
                    "last_indexed": datetime.now().isoformat(),
                })
                if ext == "xlsx":
                    meta.update({
                        "sheet_names": sheet_names,
                        "columns_by_sheet": columns_by_sheet,
                        "columns": list(set(col for cols in columns_by_sheet.values() for col in cols))
                    })
                    col_aliases = map_columns_to_concepts(meta["columns"], updated_global_aliases)
                    meta["column_aliases"] = col_aliases
                    updated_global_aliases.update(col_aliases)

                llm_meta = generate_llm_metadata(text, ext)
                meta.update(llm_meta)
                meta.update(extract_structural_metadata(text, ext))
                save_metadata(file_name, meta)

                for chunk in chunk_text(text):
                    all_chunks.append((meta, chunk))

                if meta.get("summary"):
                    embedding_cache[file_name] = get_embedding(meta["summary"])
        except Exception as e:
            st.warning(f"Failed to process {file_name}: {e}")
    else:
        if meta.get("summary"):
            for chunk in chunk_text(meta["summary"]):
                all_chunks.append((meta, chunk))
        if meta.get("embedding"):
            embedding_cache[file_name] = meta["embedding"]

    progress_bar.progress((idx + 1) / len(all_files), text=f"Indexed {idx+1}/{len(all_files)} files")
progress_bar.empty()
update_global_aliases(updated_global_aliases)

# --- Question form ---
with st.form("question_form", clear_on_submit=False):
    user_query = st.text_input("Your question:", value=st.session_state.get("last_query", ""))
    submit = st.form_submit_button("Ask")

if submit and user_query.strip():
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
    save_learned_answers(learned_answers)

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

