import streamlit as st
import os
import json
import re
from uuid import uuid4
import tempfile
import warnings
import textwrap
import time
from datetime import datetime
from openai import OpenAI
from docx import Document
import openpyxl
from pptx import Presentation
import matplotlib.pyplot as plt
import pdfplumber
import pytesseract
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import joblib  # For loading prebuilt models
from io import BytesIO
from supabase_utils import (
    list_supported_files,
    upload_file,
    download_file,
    get_file_last_modified,
    load_metadata,
    save_metadata,
    load_global_aliases,
    update_global_aliases,
    load_learned_answers,
    save_learned_answers
)
from supabase_utils import insert_embedding_chunk
import nltk
from difflib import SequenceMatcher
import traceback
import base64
import pickle
import io

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


# --- Model Config Constants ---
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_CHAT_MODEL = "gpt-4o"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

embedding_cache = {}

# --- Retry Logic for OpenAI Calls ---
def openai_with_retry(call_fn, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            return call_fn()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise e

# user query updgrade
def run_user_query(user_query, all_chunks):
    scored_chunks = []
    query_embedding = get_embedding(user_query)
    for meta, chunk in all_chunks:
        meta_text = " ".join([
            str(meta.get("title", "")).lower(),
            str(meta.get("category", "")).lower(),
            " ".join(meta.get("tags", [])),
            str(meta.get("source_file", "")).lower(),
            str(meta.get("summary", "")).lower(),
            chunk.lower()
        ])
        keyword_score = sum(word in meta_text for word in user_query.lower().split())
        emb = embedding_cache.get(meta.get("source_file"))
        semantic_score = cosine_similarity(query_embedding, emb) if emb is not None else 0.0
        hybrid_score = 0.5 * keyword_score + 0.5 * semantic_score
        scored_chunks.append((hybrid_score, meta, chunk, keyword_score, semantic_score))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = scored_chunks[:3] if scored_chunks else []

    context = "\n\n".join([chunk for _, _, chunk, _, _ in top_chunks])
    sources = [meta.get("source_file", "") for _, meta, _, _, _ in top_chunks]
    scores = [{"file": meta.get("source_file", ""), "score": score, "keyword": kw, "semantic": sem}
              for score, meta, _, kw, sem in top_chunks]

    system_prompt = (
        "You are a business analyst answering questions using internal documents (Excel, Word, PDF, PPTX). "
        "For each question, follow this reasoning chain:\n"
        "1. Identify the key data needed to answer the question.\n"
        "2. Retrieve or summarize the relevant information from the provided context.\n"
        "3. If the answer is quantitative and a chart would help, describe the chart (or provide code for matplotlib/seaborn if possible).\n"
        "4. Explain what the result or chart shows.\n"
        "5. Suggest possible causes or business insights that could explain the observed pattern, using supporting evidence from other loaded data if possible.\n"
        "6. Cite the source file(s) used for the answer.\n"
        "7. Rate your confidence in the answer (0-100) based on relevance scores and verification.\n"
        "Use only the provided context and cite sources."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"LLM answer failed: {e}"

    verification = verify_answer(answer, context, user_query)
    root_cause_analysis = mine_for_root_causes(user_query, all_chunks, top_chunks)

    st.markdown("### 📊 Predictive Modeling Options")
    predictive_model_result = None

    # --- Prebuilt Model ---
    if st.button("⚙️ Run Prebuilt Model"):
        predictive_model_result = predictive_modeling_prebuilt(user_query, top_chunks)
        st.write(predictive_model_result)

    # --- Guided Model Builder (LLM-guided pipeline) ---
    if st.button("🛠 Build and Train Model (LLM-guided)"):
        with st.spinner("Generating and training your model..."):
            result = build_and_run_model(user_query, top_chunks)
            st.write(result)

    # --- LLM-powered business logic simulation ---
    if st.button("🧠 Simulate Prediction (No Training Required)"):
        inference_result = predictive_modeling_inference(user_query, top_chunks)
        st.write(inference_result)

    st.markdown(f"**Answer:**\n\n{answer}")
    st.markdown("**Sources used:**")
    for src in sources:
        st.write(f"- {src}")
    st.markdown("**Relevance scores:**")
    for s in scores:
        st.write(f"{s['file']}: hybrid={s['score']:.2f}, keyword={s['keyword']}, semantic={s['semantic']:.2f}")
    st.markdown("**Verification:**")
    st.write(verification)
    st.markdown("**Root Cause/Data Mining Analysis:**")
    st.write(root_cause_analysis)

    top_meta = top_chunks[0][1] if top_chunks else None
    top_file = top_meta.get("source_file") if top_meta else None
    if top_file and top_file.lower().endswith('.xlsx'):
        st.markdown("---")
        st.markdown("**You can generate a chart from the top Excel file:**")
        if st.button("Show chart for top Excel file"):
            column_aliases = top_meta.get("column_aliases", {})
            excel_qa(top_file, user_query, column_aliases)

    st.session_state.last_query = user_query
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.query_log.append({
        "timestamp": datetime.now().isoformat(),
        "question": user_query,
        "files_used": sources,
        "scores": scores,
        "answer": answer,
        "verification": verification,
        "root_cause_analysis": root_cause_analysis,
        "predictive_model_result": predictive_model_result if predictive_model_result else "",
    })


# --- Supabase Storage Path for Models ---
MODELS_FOLDER = "04_Data/Models"

# --- Supabase Metadata, Alias, and Q&A Store ---
# Use load_metadata, save_metadata, load_global_aliases, update_global_aliases, load_learned_answers, save_learned_answers directly from supabase_utils

# --- Vector Validation ---
def validate_embedding(vec, expected_dim=1536):
    if vec is None or not isinstance(vec, np.ndarray) or vec.shape[0] != expected_dim:
        return np.zeros(expected_dim)
    return vec

def get_embedding(text):
    try:
        response = openai_with_retry(
            lambda: client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=text
            )
        )
        return validate_embedding(np.array(response.data[0].embedding))
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return np.zeros(1536)

def cosine_similarity(a, b):
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def detect_alias_conflicts(mapping):
    reverse = defaultdict(list)
    for col, alias in mapping.items():
        reverse[alias].append(col)
    conflicts = {alias: cols for alias, cols in reverse.items() if len(cols) > 1 and alias != "ignore"}
    return conflicts

# --- Hybrid Fuzzy+GPT Column Mapping ---
def load_concepts(path="alias_concepts.json"):
    with open(path, "r") as f:
        return json.load(f)

def extract_text_for_metadata(path, max_ocr_pages=3):
    ext = path.split(".")[-1].lower()
    try:
        file_stream = download_file(path)

        if ext == "docx":
            doc = Document(file_stream)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return text, [], {}

        elif ext == "xlsx":
            wb = openpyxl.load_workbook(file_stream, read_only=True)
            text = []
            sheet_names = []
            columns_by_sheet = {}
            for sheet in wb.worksheets:
                sheet_names.append(sheet.title)
                text.append(f"Sheet: {sheet.title}")
                headers = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
                columns_by_sheet[sheet.title] = [str(h) for h in headers if h]
                if headers:
                    text.append("Columns: " + " | ".join(columns_by_sheet[sheet.title]))
            return "\n".join(text), sheet_names, columns_by_sheet

        elif ext == "pptx":
            prs = Presentation(file_stream)
            text = "\n".join(
                shape.text for slide in prs.slides for shape in slide.shapes if shape.has_text_frame
            )
            return text, [], {}

        elif ext == "pdf":
            with pdfplumber.open(file_stream) as pdf:
                extracted_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
                if any(extracted_text):
                    return "\n".join(extracted_text), [], {}

                # No extractable text — prompt for OCR fallback
                ocr_text = []
                st.warning("No extractable text found. This PDF may require OCR.")
                if st.checkbox("Run full OCR on all pages? (May take time)"):
                    pages_to_ocr = pdf.pages
                else:
                    pages_to_ocr = pdf.pages[:max_ocr_pages]
                    st.info(f"Only scanning first {max_ocr_pages} pages with OCR...")

                for i, page in enumerate(pages_to_ocr):
                    st.write(f"OCR processing page {i+1}/{len(pages_to_ocr)}")
                    image = page.to_image(resolution=300).original
                    ocr_page = pytesseract.image_to_string(image)
                    if ocr_page:
                        ocr_text.append(ocr_page)
                return "\n".join(ocr_text), [], {}

    except Exception as e:
        st.warning(f"Failed to extract text from {path}: {e}")
    return "", [], {}

def fuzzy_match(col, concepts):
    best_match = None
    best_score = 0
    for concept in concepts:
        score = SequenceMatcher(None, col.lower(), concept.lower()).ratio()
        if score > best_score:
            best_match = concept
            best_score = score
    return best_match if best_score > 0.8 else None

def gpt_fallback_alias(col, concepts, openai_client):
    prompt = (
        f"You're mapping messy Excel headers to standardized data concepts.\n"
        f'Column: "{col}"\n'
        f"Concepts: {', '.join(concepts)}.\n"
        f'Which one best fits? If none, respond "ignore".'
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()

def map_columns_to_concepts(columns, concepts=None, use_gpt=True, openai_client=None):
    if concepts is None:
        concepts = load_concepts()
    if openai_client is None:
        raise ValueError("openai_client must be provided for GPT fallback.")

    alias_map = {}
    for col in columns:
        # Step 1: Fuzzy match
        alias = fuzzy_match(col, concepts)

        # Step 2: GPT fallback if fuzzy match fails
        if not alias and use_gpt:
            alias = gpt_fallback_alias(col, concepts, openai_client)

        # Step 3: If still nothing, ignore
        alias_map[col] = alias if alias in concepts else "ignore"

    return alias_map

def chunk_text(text, chunk_size=2000, overlap=200, max_chars=4000):
    """
    Splits text into sentence-based chunks with optional overlap and character limit.
    Returns a list of (chunk, start_idx, end_idx).
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    start_idx = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_len + len(sent) > max_chars or len(current_chunk) >= chunk_size:
            chunk_text = " ".join(current_chunk).strip()
            end_idx = start_idx + len(chunk_text)
            chunks.append((chunk_text, start_idx, end_idx))
            start_idx = end_idx
            # Overlap logic
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:]
                current_len = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_len = 0
        current_chunk.append(sent)
        current_len += len(sent)
    # Add last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        end_idx = start_idx + len(chunk_text)
        chunks.append((chunk_text, start_idx, end_idx))
    return chunks

def parse_loose_metadata(text_response):
    meta = {}
    lines = text_response.splitlines()
    for line in lines:
        if line.lower().startswith("title:"):
            meta["title"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("summary:"):
            meta["summary"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("tags:"):
            meta["tags"] = [t.strip() for t in line.split(":", 1)[1].split(",")]
        elif line.lower().startswith("category:"):
            meta["category"] = line.split(":", 1)[1].strip()
    return meta

def generate_llm_metadata(text, file_type):
    prompt = f"""
You are an expert document classification system trained to extract structured metadata from internal business documents.

Your task is to analyze the following file content and return metadata that will be used to:
- Select relevant documents in response to business questions
- Embed file context for AI search
- Categorize content for analytics and modeling

📄 File Type: {file_type}

📝 Analyze the content below (first 4000 characters):
{text[:4000]}

🎯 Return structured JSON metadata in the format:
{{
  "title": "Short descriptive title (10 words max)",
  "summary": "Detailed overview of the document content, including its business purpose, key fields, sheet/section names, and any key metrics or concepts. Use 4–6 sentences or more if needed.",
  "tags": [
    "specific keyword or metric",
    "synonym or variant if applicable",
    "e.g., 'par levels', 'usage', 'utilization'",
    "limit 7–10 tags",
    "avoid generic adjectives"
  ],
  "category": "Primary topic or department (e.g., 'supply_chain', 'finance', 'compliance')"
}}

📌 Rules:
- Do not guess or invent details not supported by content
- Tags must reflect exact terms OR common business synonyms
- Summary must be informative for AI or human previewers
- Output only valid JSON — no markdown, explanation, or comments
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()

        # Remove markdown/code block wrapper if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)

        try:
            return json.loads(raw)
        except Exception:
            return parse_loose_metadata(raw)

    except Exception as e:
        st.warning(f"LLM metadata generation failed: {e}")
        return {}

def fuzzy_match_score(word, text, threshold=0.85):
    return any(SequenceMatcher(None, word, token).ratio() >= threshold for token in text.split())

# Example usage in your scoring loop:
# keyword_score = sum(fuzzy_match_score(word, meta_text) for word in user_query.lower().split())

def extract_structural_metadata(text, file_type):
    if file_type == "docx":
        return {"section_headings": [line.strip() for line in text.splitlines() if line.strip().endswith(":") or line.strip().istitle()]}
    elif file_type == "pdf":
        return {"section_headings": [line.strip() for line in text.splitlines() if line.strip().endswith(":") or line.strip().istitle()]}
    elif file_type == "pptx":
        return {"slide_titles": [line.strip() for line in text.splitlines() if len(line.strip()) > 0]}
    return {}

def get_file_last_modified(path):
    return get_file_last_modified(path)  # Use Supabase version from supabase_utils

def verify_answer(answer, context, user_query):
    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a fact checker. Given a context and an answer, rate the factual accuracy (0-100) and list any unsupported claims."},
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}\n\nQuestion:\n{user_query}"}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Verification failed: {e}"

def mine_for_root_causes(user_query, all_chunks, top_chunks):
    mining_prompt = """
You are a senior data analyst trained to identify operational root causes across business documents such as Excel files, reports, audits, or presentations.

Your task is to analyze the provided context in response to a user’s question. Using supporting content from the data, identify possible root causes, key drivers, or anomalies. These may relate to inventory problems, demand changes, usage patterns, compliance issues, or other operational signals.

📌 Follow this step-by-step process:
1. Understand the business question.
2. Analyze patterns or outliers in the top-ranked context (recent data or closest match).
3. Compare that to broader trends in the full dataset.
4. Identify one or more likely explanations supported by the data.
5. Reference any specific fields, sheets, or files that support your findings.

💡 Be specific — cite exact values, column names, time periods, or data types if visible.

Return your analysis in this format (JSON only):
{
  "root_cause_summary": "Concise paragraph explaining the issue",
  "supporting_evidence": [
    {"file": "filename.xlsx", "reason": "Column 'usage' drops in March while 'inventory' spikes"},
    {"file": "audit_report.pdf", "reason": "Notes a stock adjustment not reflected in system"}
  ],
  "confidence_score": 0–100
}

Do not include markdown, comments, or speculative guesses without data support.
"""
    all_context = "\n\n".join([chunk for _, chunk in all_chunks])
    top_context = "\n\n".join([chunk for _, chunk in top_chunks])
    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": mining_prompt},
                {"role": "user", "content": f"Question:\n{user_query}\n\nRelevant context:\n{top_context}\n\nAll data:\n{all_context[:8000]}"}
            ],
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        # Remove markdown/code block wrapper if present (robust, regex-based)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        try:
            return json.loads(raw)
        except Exception:
            return raw  # fallback: return raw text if not valid JSON
    except Exception as e:
        return f"Root cause mining failed: {e}"

def predictive_modeling_prebuilt(user_query, top_chunks):
    # Use Supabase-native file listing
    model_files = [f for f in list_supported_files(MODELS_FOLDER) if f["name"].endswith('.pkl')]
    if not model_files:
        return "No prebuilt models found. Please build and upload a model first."
    try:
        model_file = model_files[0]
        model_path = model_file["name"]
        model_stream = download_file(model_path)
        if not isinstance(model_stream, BytesIO):
            model_stream = BytesIO(model_stream.read())
        model = joblib.load(model_stream)
        return f"Prebuilt model '{os.path.basename(model_path)}' is available. Please implement feature extraction for real predictions."
    except Exception as e:
        return f"Could not load or run prebuilt model: {e}"

def predictive_modeling_guided(user_query, top_chunks):
    modeling_prompt = (
        "You are a predictive modeling assistant. Given a business question and relevant data, "
        "suggest an appropriate predictive model, describe the features to use, and provide Python code "
        "to train and use the model (e.g., scikit-learn, XGBoost). Explain the expected results and how this helps solve the business problem."
    )
    context = "\n\n".join([chunk for _, chunk in top_chunks])
    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": modeling_prompt},
                {"role": "user", "content": f"Question:\n{user_query}\n\nRelevant data:\n{context}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Guided modeling failed: {e}"

def predictive_modeling_inference(user_query, top_chunks):
    inference_prompt = (
        "You are a business analyst. Given a question and relevant data, use statistical reasoning to simulate a prediction. "
        "Explain your reasoning and cite supporting data."
    )
    context = "\n\n".join([chunk for _, chunk in top_chunks])
    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": inference_prompt},
                {"role": "user", "content": f"Question:\n{user_query}\n\nRelevant data:\n{context}"}
            ],
            temperature = 0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM-powered inference failed: {e}"

def excel_qa(file_path, user_query, column_aliases=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re

    try:
        file_stream = download_file(file_path)
        df = pd.read_excel(file_stream)
        auto_chart = any (word in user_query.lower() for word in ["trend", "compare", "distribution", "growth", "pattern", "chart", "plot", "visual"])
        prompt = (
            f"Column aliases for this file: {json.dumps(column_aliases or {})}\n"
            f"You are a data analyst working with the following Excel file.\n"
            f"Columns: {list(df.columns)}\n"
            f"Sample data:\n{df.head(5).to_string(index=False)}\n\n"
            f"User question: {user_query}\n\n"
            "Follow this reasoning chain:\n"
            "1. Identify the key data needed to answer the question.\n"
            "2. Retrieve or summarize the relevant data.\n"
            f"3. {'Generate a chart if it would help illustrate the answer (use matplotlib/seaborn and show it).' if auto_chart else 'Generate a chart if useful.'}\n"
            "4. Explain what the result or chart shows.\n"
            "5. Suggest 1–2 possible causes or business insights that could explain the observed pattern.\n\n"
            "Return only valid Python code that uses the provided 'df' DataFrame (do NOT reload or create new data). "
            "Assign any tabular result to a variable named 'result'.\n"
            "After the code block, provide your explanation and insights."
        )

        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()
        code_match = re.search(r"```(?:python)?(.*?)```", answer, re.DOTALL)
        code = code_match.group(1).strip() if code_match else answer.strip()

        explanation = ""
        if code_match:
            explanation = answer[code_match.end():].strip()

        local_vars = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns}
        plt.clf()

        try:
            exec(code, {}, local_vars)
            if "result" in local_vars:
                st.write("✅ Result:")
                if isinstance(local_vars["result"], pd.DataFrame):
                    st.dataframe(local_vars["result"].astype(str))
                else:
                    st.write(local_vars["result"])
            st.pyplot(plt)
        except Exception as e:
            st.warning("⚠️ GPT-generated code did not execute successfully.")
            st.text(code)
            st.error(str(e))

        if explanation:
            st.markdown(f"**Explanation:** {explanation}")
    except Exception as e:
        st.error(f"Excel Q&A failed: {e}")

# --- Load global column aliases from Supabase ---
global_aliases = load_global_aliases()

# --- Place the upload expander block here, after all helper functions are defined ---
with st.expander("📁 Upload a File to Supabase"):
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "xlsx", "pptx"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        folder_map = {
            "xlsx": "02_Excel",
            "docx": "01_Docs",
            "pdf": "01_Docs",
            "pptx": "03_PPT"
        }
        ext = filename.split(".")[-1].lower()
        folder = folder_map.get(ext, "unknown")
        path = f"{folder}/{filename}"
        try:
            upload_file(path, file_bytes)
            st.success(f"✅ Uploaded to Supabase at `{path}`")
            # --- Immediately extract and save metadata for the uploaded file ---
            # Always reload global_aliases in case of concurrent edits
            global_aliases = load_global_aliases()
            if ext == "xlsx":
                text, sheet_names, columns_by_sheet = extract_text_for_metadata(path)
            else:
                text = extract_text_for_metadata(path)
                sheet_names, columns_by_sheet = [], {}

            if isinstance(text, tuple):
                text = text[0]

            if text and text.strip():
                gpt_meta = generate_llm_metadata(text, ext)
                gpt_meta["source_file"] = path
                gpt_meta["last_indexed"] = datetime.now().isoformat()
                gpt_meta["author"] = st.secrets.get("user_email", "system")
                gpt_meta["file_type"] = ext
                gpt_meta["file_size"] = len(file_bytes)
                gpt_meta["last_modified"] = datetime.now().isoformat()

                if ext == "xlsx":
                    gpt_meta["sheet_names"] = sheet_names
                    gpt_meta["columns_by_sheet"] = columns_by_sheet
                    all_columns = []
                    if columns_by_sheet:
                        for cols in columns_by_sheet.values():
                            all_columns.extend(cols)
                    gpt_meta["columns"] = list(set(all_columns)) if all_columns else []
                    if gpt_meta["columns"]:
                        # Use hybrid fuzzy+GPT mapping
                        alias_concepts = load_concepts()
                        column_aliases = map_columns_to_concepts(gpt_meta["columns"], alias_concepts, use_gpt=True, openai_client=client)
                        gpt_meta["column_aliases"] = column_aliases
                        # Only update global aliases with valid concepts
                        valid_aliases = {k: v for k, v in column_aliases.items() if v != "ignore"}
                        update_global_aliases(valid_aliases)

                gpt_meta.update(extract_structural_metadata(text, ext))
                save_metadata(path, gpt_meta)
                st.success(f"📝 Metadata extracted and saved for `{path}`.")

                # --- Insert chunk-level embeddings into embedding index (robust, both gpt_meta and meta for file_id) ---
                if text.strip():
                    chunks = chunk_text(text)  # (chunk_text, start_idx, end_idx)
                    # meta is not defined in this scope, so set to None for safety
                    meta = None
                    file_id = gpt_meta.get("id") or (meta.get("id") if meta else None)

                    if file_id:
                        for chunk_text_content, start, end in chunks:
                            try:
                                vector = get_embedding(chunk_text_content)
                                chunk_id = str(uuid4())
                                token_count = len(chunk_text_content.split())
                                insert_embedding_chunk(
                                    file_id=file_id,
                                    chunk_id=chunk_id,
                                    chunk_text=chunk_text_content,
                                    embedding=vector.tolist(),
                                    token_count=token_count
                                )
                            except Exception as e:
                                st.warning(f"⚠️ Embedding failed for a chunk: {e}")
                    else:
                        st.warning("⚠️ No file_id found for embedding_index insert.")

                # --- Insert chunk-level embeddings into embedding index ---
                for chunk_text_content, start, end in chunk_text(text):
                    try:
                        vector = get_embedding(chunk_text_content)
                        file_id = gpt_meta.get("id")  # Must exist in metadata
                        chunk_id = str(uuid4())
                        token_count = len(chunk_text_content.split())

                        # Insert into embedding_index table
                        insert_embedding_chunk(
                            file_id=file_id,
                            chunk_id=chunk_id,
                            chunk_text=chunk_text_content,
                            embedding=vector.tolist(),
                            token_count=token_count
                        )
                    except Exception as e:
                        st.warning(f"Embedding chunk failed: {e}")
            else:
                st.warning(f"No extractable text found in `{path}` for metadata generation.")
                st.info(f"Debug: text extraction result: {text}")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")
            st.info(f"Debug: Exception details: {traceback.format_exc()}")

# --- Streamlit UI for editing column aliases ---
if st.checkbox("🔧 Edit Column Alias Mappings"):
    current_aliases = load_global_aliases()
    editable_aliases = st.experimental_data_editor(current_aliases, num_rows="dynamic", key="alias_editor")
    if st.button("💾 Save Updated Aliases"):
        update_global_aliases(editable_aliases)
        st.success("Aliases updated successfully.")

# --- Gather all files from Supabase for Q&A and modeling ---
# List all supported files in Supabase storage, EXCLUDING raw data folder
all_files = [
    f
    for f in list_supported_files()
    if not f["name"].startswith("Raw_Data")
]

all_chunks = []
updated_global_aliases = global_aliases.copy()
progress_bar = st.progress(0, text="Indexing files and building metadata...")

for idx, file in enumerate(all_files):
    file_name = file["name"]
    ext = file_name.lower().split('.')[-1]

    file_last_modified = get_file_last_modified(file_name)
    meta = load_metadata(file_name) or {"source_file": file_name}
    needs_index = True

    # --- Efficient re-indexing: skip if file hasn't changed since last_indexed
    if meta.get("last_indexed") and file_last_modified:
        try:
            # file_last_modified is ISO string from Drive, meta["last_indexed"] is ISO string
            from dateutil.parser import isoparse
            if isoparse(file_last_modified) <= isoparse(meta["last_indexed"]):
                needs_index = False
        except Exception:
            pass

    if needs_index:
        try:
            if ext == "xlsx":
                text, sheet_names, columns_by_sheet = extract_text_for_metadata(file_name)
            else:
                text = extract_text_for_metadata(file_name)
                sheet_names, columns_by_sheet = [], {}

            if isinstance(text, tuple):
                text = text[0]

            if text.strip():

                # --- Generate LLM metadata and append audit fields ---
                gpt_meta = generate_llm_metadata(text, ext)
                gpt_meta["source_file"] = file_name  # The file name in Supabase
                gpt_meta["last_indexed"] = datetime.now().isoformat()
                gpt_meta["author"] = st.secrets.get("user_email", "system")

                # Add additional metadata if needed
                gpt_meta["file_type"] = ext
                gpt_meta["file_size"] = file.get("size", None)
                gpt_meta["last_modified"] = file_last_modified

                # Excel-specific
                if ext == "xlsx":
                    gpt_meta["sheet_names"] = sheet_names
                    gpt_meta["columns_by_sheet"] = columns_by_sheet
                    all_columns = []
                    if columns_by_sheet:
                        for cols in columns_by_sheet.values():
                            all_columns.extend(cols)
                    gpt_meta["columns"] = list(set(all_columns)) if all_columns else []
                    if gpt_meta["columns"]:
                        alias_concepts = load_concepts()
                        column_aliases = map_columns_to_concepts(gpt_meta["columns"], alias_concepts, use_gpt=True, openai_client=client)
                        gpt_meta["column_aliases"] = column_aliases
                        valid_aliases = {k: v for k, v in column_aliases.items() if v != "ignore"}
                        updated_global_aliases.update(valid_aliases)

                # Structural metadata for all files
                gpt_meta.update(extract_structural_metadata(text, ext))

                gpt_meta = save_metadata(file_name, gpt_meta) or gpt_meta

                st.write("🧠 save_metadata() returned:", gpt_meta)
                st.write("🧠 gpt_meta type:", type(gpt_meta))
                if isinstance(gpt_meta, dict):
                    st.write("✅ file_id from metadata:", gpt_meta.get("id"))
                else:
                    st.error("❌ gpt_meta is not a dict. Embedding will fail.")


# Safely extract file_id only if gpt_meta is a dict and contains 'id'
                file_id = gpt_meta.get("id") if isinstance(gpt_meta, dict) else None
                st.write("✅ file_id resolved:", file_id)

                # Chunk and embed full text
                if file_id:
                    chunks = chunk_text(text)
                    for idx, (chunk_text_content, start, end) in enumerate(chunks):
                        try:
                            vector = get_embedding(chunk_text_content)
                            chunk_id = str(uuid4())
                            token_count = len(chunk_text_content.split())

                            # Add to Supabase
                            insert_embedding_chunk(
                                file_id=file_id,
                                chunk_id=chunk_id,
                                chunk_text=chunk_text_content,
                                embedding=vector.tolist(),
                                token_count=token_count
                            )

                            # Also add to in-memory chunks
                            all_chunks.append((gpt_meta, chunk_text_content))
                        except Exception as e:
                            st.warning(f"⚠️ Embedding chunk failed: {e}")
                else:
                    st.warning(f"❌ No file_id found in metadata for {file_name} — skipping embedding.")

        except Exception as e:
            st.warning(f"⚠️ Failed to process {file_name}: {e}")
    else:
        # Use existing metadata and skip re-indexing
        if "summary" in meta:
            for chunk in chunk_text(meta["summary"]):
                all_chunks.append((meta, chunk))
        if "embedding" in meta and meta["embedding"]:
            embedding_cache[file_name] = np.array(meta["embedding"])

    progress_bar.progress((idx + 1) / len(all_files), text=f"Indexed {idx+1}/{len(all_files)} files")
progress_bar.empty()
update_global_aliases(updated_global_aliases)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_excel" not in st.session_state:
    st.session_state.last_excel = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "query_log" not in st.session_state:
    st.session_state.query_log = []

st.header("Ask a question about your knowledge base")

user_query = st.text_input("Your question:", value=st.session_state.get("last_query", ""), key="query_input")

submit = st.button("Ask")


if submit or (user_query and user_query != st.session_state.get("last_query")):
    run_user_query(user_query, all_chunks)

with st.expander("Show Query Log"):
    for entry in st.session_state.query_log:
        st.write(entry)

with st.expander("Show Chat History"):
    for msg in st.session_state.chat_history:
        st.write(f"{msg['role'].capitalize()}: {msg['content']}")

# --- GPT Prompt & Model Generation ---
def get_model_prompt_response(user_query, context_df_sample):
    modeling_prompt = f"""
You are a senior machine learning engineer. You are given a business question and structured tabular data from Excel files.

Your task is to:
1. Identify the prediction goal (e.g., regression, classification, forecasting).
2. Choose the target variable and relevant features from the dataset.
3. Generate a robust Python model training pipeline using scikit-learn or XGBoost.
4. Include data cleaning, encoding, train/test split, training, evaluation, and prediction on test data.

Use the DataFrame variable `df` as your data.

🧾 Sample of the data:
{context_df_sample.to_string(index=False)}

📄 User Question:
{user_query}

Return valid JSON:
{{
  "model_type": "regression | classification | forecasting | anomaly_detection",
  "target_variable": "string",
  "features": ["col1", "col2", ...],
  "model_code": "FULL Python code as string that uses df",
  "description": "Explanation of the model design"
}}

Do not include markdown or comments. JSON only.
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": modeling_prompt}],
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw.strip("`").replace("json", "").strip()
        return json.loads(raw)
    except Exception as e:
        st.warning("❌ GPT model generation failed.")
        st.text(traceback.format_exc())
        return None

# --- Execute Model Code Safely ---
def run_model_code(model_code, df):
    local_vars = {"df": df.copy()}
    try:
        exec(model_code, {}, local_vars)
        return local_vars
    except Exception as e:
        st.error("❌ Failed to execute generated model code.")
        st.text(model_code)
        st.text(traceback.format_exc())
        return None

# --- Save Model as .pkl ---


def save_model(local_vars, model_name="auto_model.pkl"):
    model = local_vars.get("model", None)
    if not model:
        st.warning("⚠️ No model object found to save.")
        return

    try:
        # Serialize model to bytes
        model_bytes = pickle.dumps(model)
        buffer = BytesIO(model_bytes)
        buffer.seek(0)

        # --- Save model to Supabase or local storage ---
        # Correct usage: path first, then bytes
        upload_file(f"{MODELS_FOLDER}/{model_name}", buffer.read())
        st.success(f"✅ Model saved to Supabase as {model_name}")
    except Exception as e:
        st.error(f"❌ Failed to save model: {e}")

# --- MAIN: Guided Model Builder ---
def build_and_run_model(user_query, top_chunks, target_column=None):
    try:
        # Build sample DataFrame from context
        sample_df = None
        for meta, chunk in top_chunks:
            file = meta.get("source_file", "")
            if file.endswith(".xlsx"):
                file_stream = download_file(file)
                sample_df = pd.read_excel(file_stream)
                break
        if sample_df is None:
            return "⚠️ No usable Excel data found for modeling."

        # Get model generation response from GPT
        result = get_model_prompt_response(user_query, sample_df.head(5))
        if not result:
            return "❌ Model generation failed."

        model_code = result.get("model_code", "")
        st.subheader("🧠 GPT-Generated Model")
        st.code(model_code, language="python")

        local_vars = run_model_code(model_code, sample_df)
        if not local_vars:
            return "⚠️ Code execution failed."

        save_model(local_vars)

        # Show predictions if available
        if "y_pred" in local_vars:
            st.subheader("📈 Model Predictions")
            st.dataframe(pd.DataFrame(local_vars["y_pred"], columns=["Prediction"]))
    except Exception as e:
        st.error(f"Model building/running failed: {e}")

def structured_data_qa(user_query, top_chunks):
    excel_dfs = []
    doc_texts = []

    # --- Separate structured Excel data and unstructured document chunks ---
    for meta, chunk in top_chunks:
        file = meta.get("source_file", "")
        ext = file.lower().split('.')[-1]
        if ext == "xlsx":
            try:
                file_stream = download_file(file)
                xl = pd.ExcelFile(file_stream)
                for sheet_name in xl.sheet_names:
                    df = xl.parse(sheet_name)
                    excel_dfs.append((file, sheet_name, df))
            except Exception as e:
                st.warning(f"Failed to read Excel file: {file}: {e}")
        elif ext in ["pdf", "docx", "pptx"]:
            doc_texts.append(chunk)

    # --- Prepare structured Excel sample prompt ---
    excel_context = []
    for file, sheet, df in excel_dfs:
        context = f"File: {file}, Sheet: {sheet}\n(df = this sheet)\n{df.head(5).to_string(index=False)}"
        excel_context.append(context)

    excel_prompt = "\n\n".join(excel_context)
    doc_context = "\n\n".join(doc_texts)

    # --- Build GPT prompt ---
    prompt = f"""
You are a business data analyst. The user has asked the following question:

🧠 Question: {user_query}

You are given two types of information:
1. Structured Excel data across multiple files and sheets. For each sample, use the variable `df` as shown.
2. Unstructured document content from PDFs, Word files, and presentations.

📊 Excel Data Sample:
{excel_prompt}

📄 Document Content:
{doc_context[:3000]}

🎯 Your task is to analyze both sources and return a dual-section answer. If helpful, include Python code using the provided 'df' variable for visuals.

### Excel Data Insights:
- Analyze key trends, distributions, comparisons, or anomalies.
- Use tabular summaries or charts if useful.
- Base all findings on the provided data.

### Document-Based Insights:
- Provide any root causes, definitions, procedures, or context.
- Reference specific language if relevant.

Be specific and avoid generic responses. Return only markdown-formatted content with labeled sections.
"""

    # --- Call GPT to analyze both structured and unstructured content ---
    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior data analyst responding in markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        reply = response.choices[0].message.content.strip()

        # --- Extract code block(s) if any, and explanation ---
        code_blocks = re.findall(r"```(?:python)?(.*?)```", reply, re.DOTALL)
        code = code_blocks[0].strip() if code_blocks else None
        explanation = reply
        if code:
            # Remove the code block from the explanation for clarity
            explanation = reply.replace(f"```python\n{code}\n```", "").replace(f"```\n{code}\n```", "").strip()

        # --- Display insights first ---
        st.markdown("### 🤖 GPT Analysis")
        st.markdown(explanation)

        # --- Execute GPT-generated charting code and display/download chart ---
        if code:
            plt.clf()
            local_vars = {"pd": pd, "plt": plt, "sns": sns}
            # If only one Excel sample, provide df for code compatibility
            if len(excel_dfs) == 1:
                local_vars["df"] = excel_dfs[0][2]
            try:
                exec(code, {}, local_vars)
                chart_buffer = io.BytesIO()
                plt.savefig(chart_buffer, format="png", bbox_inches="tight")
                chart_buffer.seek(0)
                st.pyplot(plt)
                st.download_button(
                    label="📥 Download Chart as PNG",
                    data=chart_buffer,
                    file_name="chart_output.png",
                    mime="image/png"
                )
                if "result" in local_vars:
                    st.dataframe(local_vars["result"])
            except Exception as e:
                st.error("⚠️ Error running GPT-generated chart code")
                st.code(code, language="python")
                st.text(str(e))

    except Exception as e:
        st.error(f"Structured Q&A failed: {e}")

def run_user_query(user_query, all_chunks):
    from difflib import SequenceMatcher

    def find_similar_learned_answer(query, learned_answers, threshold=0.85):
        def similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()
        for past_question, entry in learned_answers.items():
            if similarity(query, past_question) >= threshold:
                return entry
        return None


    learned_answers = load_learned_answers()
    cached = find_similar_learned_answer(user_query, learned_answers)
    if cached:
        st.success("✅ Reusing a previously learned answer")
        st.markdown("**Answer:**")
        st.markdown(cached["answer"])
        st.markdown("**Sources used:**")
        for src in cached.get("files_used", []):
            st.write(f"- {src}")
        return

    scored_chunks = []
    query_embedding = get_embedding(user_query)
    for meta, chunk in all_chunks:
        meta_text = " ".join([
            str(meta.get("title", "")).lower(),
            str(meta.get("category", "")).lower(),
            " ".join(meta.get("tags", [])),
            str(meta.get("source_file", "")).lower(),
            str(meta.get("summary", "")).lower(),
            chunk.lower()
        ])
        keyword_score = sum(word in meta_text for word in user_query.lower().split())
        emb = embedding_cache.get(meta.get("source_file"))
        semantic_score = cosine_similarity(query_embedding, emb) if emb is not None else 0.0
        hybrid_score = 0.5 * keyword_score + 0.5 * semantic_score
        scored_chunks.append((hybrid_score, meta, chunk, keyword_score, semantic_score))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = scored_chunks[:3] if scored_chunks else []

    context = "\n\n".join([chunk for _, _, chunk, _, _ in top_chunks])
    sources = [meta.get("source_file", "") for _, meta, _, _, _ in top_chunks]
    scores = [{"file": meta.get("source_file", ""), "score": score, "keyword": kw, "semantic": sem}
              for score, meta, _, kw, sem in top_chunks]

    system_prompt = (
        "You are a business analyst answering questions using internal documents (Excel, Word, PDF, PPTX). "
        "For each question, follow this reasoning chain:\n"
        "1. Identify the key data needed to answer the question.\n"
        "2. Retrieve or summarize the relevant information from the provided context.\n"
        "3. If the answer is quantitative and a chart would help, describe the chart (or provide code for matplotlib/seaborn if possible).\n"
        "4. Explain what the result or chart shows.\n"
        "5. Suggest possible causes or business insights that could explain the observed pattern, using supporting evidence from other loaded data if possible.\n"
        "6. Cite the source file(s) used for the answer.\n"
        "7. Rate your confidence in the answer (0-100) based on relevance scores and verification.\n"
        "Use only the provided context and cite sources."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"LLM answer failed: {e}"

    verification = verify_answer(answer, context, user_query)
    root_cause_analysis = mine_for_root_causes(user_query, all_chunks, top_chunks)

    st.markdown("### 📊 Predictive Modeling Options")
    predictive_model_result = None

    # --- Prebuilt Model ---
    if st.button("⚙️ Run Prebuilt Model"):
        predictive_model_result = predictive_modeling_prebuilt(user_query, top_chunks)
        st.write(predictive_model_result)

    # --- Guided Model Builder (LLM-guided pipeline) ---
    if st.button("🛠 Build and Train Model (LLM-guided)"):
        with st.spinner("Generating and training your model..."):
            result = build_and_run_model(user_query, top_chunks)
            st.write(result)

    # --- LLM-powered business logic simulation ---
    if st.button("🧠 Simulate Prediction (No Training Required)"):
        inference_result = predictive_modeling_inference(user_query, top_chunks)
        st.write(inference_result)

    st.markdown(f"**Answer:**\n\n{answer}")
    st.markdown("**Sources used:**")
    for src in sources:
        st.write(f"- {src}")
    st.markdown("**Relevance scores:**")
    for s in scores:
        st.write(f"{s['file']}: hybrid={s['score']:.2f}, keyword={s['keyword']}, semantic={s['semantic']:.2f}")
    st.markdown("**Verification:**")
    st.write(verification)
    st.markdown("**Root Cause/Data Mining Analysis:**")
    st.write(root_cause_analysis)

    top_meta = top_chunks[0][1] if top_chunks else None
    top_file = top_meta.get("source_file") if top_meta else None
    if top_file and top_file.lower().endswith('.xlsx'):
        st.markdown("---")
        st.markdown("**You can generate a chart from the top Excel file:**")
        if st.button("Show chart for top Excel file"):
            column_aliases = top_meta.get("column_aliases", {})
            excel_qa(top_file, user_query, column_aliases)

    st.session_state.last_query = user_query
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.query_log.append({
        "timestamp": datetime.now().isoformat(),
        "question": user_query,
        "files_used": sources,
        "scores": scores,
        "answer": answer,
        "verification": verification,
        "root_cause_analysis": root_cause_analysis,
        "predictive_model_result": predictive_model_result if predictive_model_result else "",
    })

with st.expander("Show Query Log"):
    for entry in st.session_state.query_log:
        st.write(entry)

with st.expander("Show Chat History"):
    for msg in st.session_state.chat_history:
        st.write(f"{msg['role'].capitalize()}: {msg['content']}")

