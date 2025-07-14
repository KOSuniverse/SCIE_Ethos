import streamlit as st
import os
import json
from datetime import datetime
from openai import OpenAI
from docx import Document
import openpyxl
from pptx import Presentation
import pdfplumber
import pytesseract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PROJECT_ROOT = "./Project_Root"
GLOBAL_ALIAS_PATH = os.path.join(PROJECT_ROOT, "global_column_aliases.json")

embedding_cache = {}

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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

def profile_file_usage(file_path, tags):
    pass

def profile_excel_file(file_path):
    pass

def extract_text_for_metadata(path, max_ocr_pages=5):
    if path.endswith(".docx"):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif path.endswith(".xlsx"):
        wb = openpyxl.load_workbook(path, read_only=True)
        text = []
        sheet_names = []
        columns_by_sheet = {}
        for sheet in wb.worksheets:
            sheet_names.append(sheet.title)
            text.append(f"Sheet: {sheet.title}")
            headers = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
            columns_by_sheet[sheet.title] = [str(h) for h in headers if h]
            if headers:
                text.append("Columns: " + " | ".join([str(h) for h in headers if h]))
        return "\n".join(text), sheet_names, columns_by_sheet
    elif path.endswith(".pptx"):
        prs = Presentation(path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text.append(shape.text)
        return "\n".join(text)
    elif path.endswith(".pdf"):
        cache_dir = os.path.join(os.path.dirname(path), "_ocr_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        if any(t.strip() for t in text):
            result = "\n".join(text)
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(result)
            return result
        ocr_text = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_ocr_pages:
                    break
                st.write(f"OCR processing page {i+1}/{min(len(pdf.pages), max_ocr_pages)}")
                image = page.to_image(resolution=300).original
                page_ocr = pytesseract.image_to_string(image)
                if page_ocr:
                    ocr_text.append(page_ocr)
        result = "\n".join(ocr_text)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(result)
        return result
    return ""

def find_all_files(base_dir, extensions=(".pdf", ".docx", ".pptx", ".xlsx")):
    file_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(extensions):
                file_list.append(os.path.join(root, file))
    return file_list

def chunk_text(text, chunk_size=2000, overlap=200):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def get_relevant_chunks(query, all_chunks, top_k=10):
    query_lower = query.lower()
    show_all = any(kw in query_lower for kw in ["all data", "all rows", "full table", "everything", "entire"])
    query_words = query_lower.split()
    scored = []
    for meta, chunk in all_chunks:
        category_value = meta.get("category", "")
        category_words = []
        if category_value:
            if isinstance(category_value, list):
                for cat in category_value:
                    category_words.extend([w.strip().lower() for w in str(cat).replace(",", " ").split() if w.strip()])
            else:
                category_words = [w.strip().lower() for w in str(category_value).replace(",", " ").split() if w.strip()]
        tag_value = meta.get("tags", [])
        tag_words = []
        if tag_value:
            if isinstance(tag_value, list):
                tag_words = [w.strip().lower() for w in tag_value if w.strip()]
            else:
                tag_words = [w.strip().lower() for w in str(tag_value).replace(",", " ").split() if w.strip()]
        meta_text = " ".join([
            str(meta.get("title", "")).lower(),
            " ".join(category_words),
            " ".join(tag_words),
            str(meta.get("source_file", "")).lower(),
            " ".join(meta.get("sheet_names", [])).lower(),
            " ".join(meta.get("columns", [])).lower(),
            str(meta.get("summary", "")).lower()
        ]).lower()
        chunk_text_l = chunk.lower()
        score = sum(word in chunk_text_l or word in meta_text for word in query_words)
        scored.append((score, meta, chunk))
    scored = sorted(scored, reverse=True, key=lambda x: x[0])
    if show_all:
        return [(meta, chunk) for score, meta, chunk in scored if score > 0] or scored
    else:
        return [(meta, chunk) for score, meta, chunk in scored[:top_k] if score > 0] or scored[:top_k]

def auto_generate_metadata(prompt_text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a metadata assistant. Generate structured metadata for internal business documents.\n"
                    "Return ONLY valid JSON with the following fields :\n"
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

def excel_qa(file_path, user_query, column_aliases=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re

    df = pd.read_excel(file_path)
    auto_chart = any(word in user_query.lower() for word in ["trend", "compare", "distribution", "growth", "pattern", "chart", "plot", "visual"])
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
        model="gpt-4o",
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
                st.dataframe(local_vars["result"])
            else:
                st.write(local_vars["result"])
        st.pyplot(plt)
    except Exception as e:
        st.warning("⚠️ GPT-generated code did not execute successfully.")
        st.text(code)
        st.error(str(e))

    if explanation:
        st.markdown(f"**Explanation:** {explanation}")

# --- Load global column aliases ---
if os.path.exists(GLOBAL_ALIAS_PATH):
    with open(GLOBAL_ALIAS_PATH, "r", encoding="utf-8") as f:
        global_aliases = json.load(f)
else:
    global_aliases = {}

# --- Gather all chunks from all files ---
all_files = find_all_files(PROJECT_ROOT)
all_chunks = []
updated_global_aliases = global_aliases.copy()
for file_path in all_files:
    ext = os.path.splitext(file_path)[1].lower()
    max_ocr_pages = 5 if ext == ".pdf" else 0
    if ext == ".xlsx":
        text, sheet_names, columns_by_sheet = extract_text_for_metadata(file_path, max_ocr_pages=max_ocr_pages)
        profile_excel_file(file_path)
    else:
        text = extract_text_for_metadata(file_path, max_ocr_pages=max_ocr_pages)
        sheet_names, columns_by_sheet = [], {}
    if text.strip():
        meta = {"source_file": os.path.basename(file_path)}
        if ext == ".xlsx":
            meta["sheet_names"] = sheet_names
            meta["columns_by_sheet"] = columns_by_sheet
            all_columns = []
            for cols in columns_by_sheet.values():
                all_columns.extend(cols)
            meta["columns"] = list(set(all_columns))
            column_aliases = map_columns_to_concepts(meta["columns"], updated_global_aliases)
            meta["column_aliases"] = column_aliases
            updated_global_aliases.update(column_aliases)
        meta_path = os.path.join(os.path.dirname(file_path), "_metadata", os.path.splitext(os.path.basename(file_path))[0] + ".json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as jf:
                    meta.update(json.load(jf))
            except Exception:
                pass
        else:
            try:
                metadata = auto_generate_metadata(text)
                meta.update(metadata)
                meta_dir = os.path.join(os.path.dirname(file_path), "_metadata")
                os.makedirs(meta_dir, exist_ok=True)
                with open(meta_path, "w") as jf:
                    json.dump(metadata, jf, indent=2)
            except Exception:
                pass
        for chunk in chunk_text(text):
            all_chunks.append((meta, chunk))
        embedding = None
        if "embedding" in meta and meta["embedding"]:
            embedding = np.array(meta["embedding"])
        elif meta.get("summary", ""):
            try:
                embedding = get_embedding(meta["summary"])
            except Exception as e:
                print(f"Embedding failed for {file_path}: {e}")
        if embedding is not None:
            embedding_cache[file_path] = embedding

with open(GLOBAL_ALIAS_PATH, "w") as f:
    json.dump(updated_global_aliases, f, indent=2)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_excel" not in st.session_state:
    st.session_state.last_excel = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "query_log" not in st.session_state:
    st.session_state.query_log = []

st.header("Ask a question about your knowledge base")

with st.form("question_form", clear_on_submit=False):
    user_query = st.text_input("Your question:", value=st.session_state.get("last_query", ""))
    submit = st.form_submit_button("Ask")

run_excel_qa = submit or (selected_excel != st.session_state.get("last_excel")) or (user_query != st.session_state.get("last_query"))

user_query = user_query or ""
query_words = [w.strip().lower() for w in user_query.split()]
matching_excels = []
excel_scores = []
for file_path in all_files:
    if file_path.endswith('.xlsx'):
        meta_path = os.path.join(
            os.path.dirname(file_path),
            "_metadata",
            os.path.splitext(os.path.basename(file_path))[0] + ".json"
        )
        meta = {"source_file": os.path.basename(file_path)}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as jf:
                    meta.update(json.load(jf))
            except Exception:
                pass
        category_value = meta.get("category", "")
        category_words = []
        if category_value:
            if isinstance(category_value, list):
                for cat in category_value:
                    category_words.extend([w.strip().lower() for w in str(cat).replace(",", " ").split() if w.strip()])
            else:
                category_words = [w.strip().lower() for w in str(category_value).replace(",", " ").split() if w.strip()]
        tag_value = meta.get("tags", [])
        tag_words = []
        if tag_value:
            if isinstance(tag_value, list):
                tag_words = [w.strip().lower() for w in tag_value if w.strip()]
            else:
                tag_words = [w.strip().lower() for w in str(tag_value).replace(",", " ").split() if w.strip()]
        sheet_names = meta.get("sheet_names", [])
        columns = meta.get("columns", [])
        sample_values = []
        try:
            df = pd.read_excel(file_path, nrows=1)
            for val in df.iloc[0].values:
                if pd.notnull(val):
                    sample_values.append(str(val).strip().lower())
        except Exception:
            pass
        meta_text = " ".join([
            str(meta.get("title", "")).lower(),
            " ".join(category_words),
            " ".join(tag_words),
            str(meta.get("source_file", "")).lower(),
            " ".join(sheet_names).lower() if isinstance(sheet_names, list) else str(sheet_names).lower(),
            " ".join(columns).lower() if isinstance(columns, list) else str(columns).lower(),
            str(meta.get("summary", "")).lower(),
            " ".join(sample_values)
        ])
        score = sum(word in meta_text for word in query_words)
        if score > 0:
            matching_excels.append(file_path)
            excel_scores.append(score)

query_embedding = get_embedding(user_query) if user_query else None
semantic_scores = []
for file_path in all_files:
    emb = embedding_cache.get(file_path)
    if query_embedding is not None and emb is not None:
        score = cosine_similarity(query_embedding, emb)
    else:
        score = 0.0
    semantic_scores.append((file_path, score))

hybrid_scores = []
for file_path in all_files:
    if file_path in matching_excels:
        idx = matching_excels.index(file_path)
        keyword_score = excel_scores[idx]
    else:
        keyword_score = 0
    semantic_score = dict(semantic_scores).get(file_path, 0)
    hybrid_score = 0.5 * keyword_score + 0.5 * semantic_score
    hybrid_scores.append((file_path, hybrid_score))

hybrid_scores.sort(key=lambda x: x[1], reverse=True)
top_files = [fp for fp, score in hybrid_scores if score > 0]

N = 3
top_files_for_context = [fp for fp, score in hybrid_scores[:N] if score > 0]
context_chunks = []
for fp in top_files_for_context:
    context_chunks.extend([chunk for meta, chunk in all_chunks if meta.get("source_file") == os.path.basename(fp)])
context = "\n\n".join(context_chunks[:4000])

selected_excel = None
if top_files:
    selected_excel = st.selectbox(
        "Select an Excel file to answer your question:",
        top_files,
        index=0,
        key="excel_select"
    )
    st.info(f"Matching Excel files: {[os.path.basename(f) for f in matching_excels]}")

    run_excel_qa = (
        st.button("Ask") or
        selected_excel != st.session_state.get("last_excel") or
        user_query != st.session_state.get("last_query")
    )
    if run_excel_qa and selected_excel and user_query:
        meta_path = os.path.join(
            os.path.dirname(selected_excel),
            "_metadata",
            os.path.splitext(os.path.basename(selected_excel))[0] + ".json"
        )
        column_aliases = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as jf:
                    meta_excel = json.load(jf)
                    column_aliases = meta_excel.get("column_aliases", {})
            except Exception:
                pass
        excel_qa(selected_excel, user_query, column_aliases)
        st.session_state.last_excel = selected_excel
        st.session_state.last_query = user_query
        profile_file_usage(selected_excel, meta.get("tags", []))
        if st.session_state.chat_history == [] or st.session_state.chat_history[-1]["content"] != user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": "Chart or result generated above."})
        try:
            excel_file = selected_excel
            excel_sheets = list(pd.ExcelFile(selected_excel).sheet_names)
        except Exception:
            excel_file = selected_excel
            excel_sheets = []
        st.session_state.query_log.append({
            "timestamp": datetime.now().isoformat(),
            "question": user_query,
            "file": excel_file,
            "excel_sheets": excel_sheets,
            "chart_generated": True,
            "reasoning": "Chart or result generated above."
        })
else:
    system_prompt = (
        "You are a helpful assistant answering questions based on internal business documents. "
        "For each question, follow this reasoning chain:\n"
        "1. Identify the key data needed to answer the question.\n"
        "2. Retrieve or summarize the relevant information from the provided context.\n"
        "3. Generate a chart if it would help illustrate the answer (return a valid Chart.js config as JSON if possible).\n"
        "4. Explain what the result or chart shows.\n"
        "5. Suggest 1–2 possible causes or business insights that could explain the observed pattern.\n"
        "Use the provided context to answer as best you can."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ],
        temperature=0.3
    )
    answer = response.choices[0].message.content.strip()
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.session_state.query_log.append({
        "timestamp": datetime.now().isoformat(),
        "question": user_query,
        "file": None,
        "excel_sheets": [],
        "chart_generated": False,
        "reasoning": answer
    })

    try:
        chart_data = json.loads(answer)
        if "chart_type" in chart_data and "x" in chart_data and "y" in chart_data:
            st.write(chart_data.get("title", "Chart"))
            if chart_data["chart_type"] == "bar":
                df = pd.DataFrame({"x": chart_data["x"], "y": chart_data["y"]})
                st.bar_chart(df.set_index("x"))
        else:
            st.write(answer)
    except Exception:
        st.write(answer)

    with st.expander("Show supporting context"):
        for fp in top_files_for_context:
            meta_path = os.path.join(
                os.path.dirname(fp),
                "_metadata",
                os.path.splitext(os.path.basename(fp))[0] + ".json"
            )
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as jf:
                        meta.update(json.load(jf))
                except Exception:
                    pass
            st.write(f"**Source:** {meta.get('title', meta.get('source_file', os.path.basename(fp)))}")
            st.write(f"**Category:** {meta.get('category', '')}")
            st.write(f"**Tags:** {', '.join(meta.get('tags', []))}")
            st.write(f"**Summary:** {meta.get('summary', '')}")
            file_chunks = [chunk for m, chunk in all_chunks if m.get("source_file") == os.path.basename(fp)]
            if file_chunks:
                st.write(file_chunks[0][:500] + ("..." if len(file_chunks[0]) > 500 else ""))
            st.write("---")

with st.expander("Show Query Log"):
    for entry in st.session_state.query_log:
        st.write(entry)

with st.container():
    st.markdown(
        """
        <style>
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 1em;
            background: #f7f7f9;
            border-radius: 8px;
            border: 1px solid #eee;
        }
        .user-msg {
            background: #d1e7dd;
            padding: 0.5em 1em;
            border-radius: 12px;
            margin-bottom: 0.5em;
            align-self: flex-end;
            max-width: 80%;
        }
        .assistant-msg {
            background: #f8d7da;
            padding: 0.5em 1em;
            border-radius: 12px;
            margin-bottom: 0.5em;
            align-self: flex-start;
            max-width: 80%;
        }
        </style>
        <div class="chat-container">
        """,
        unsafe_allow_html=True,
    )
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg"><b>You:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-msg"><b>Assistant:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

for file_path in all_files:
    meta_path = os.path.join(os.path.dirname(file_path), "_metadata", os.path.splitext(os.path.basename(file_path))[0] + ".json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            json_data = f.read()
        st.download_button(
            label=f"Download metadata for {os.path.basename(file_path)}",
            data=json_data,
            file_name=os.path.basename(meta_path),
            mime="application/json"
        )





