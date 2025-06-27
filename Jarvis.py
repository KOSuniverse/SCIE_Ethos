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

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

PROJECT_ROOT = r"G:\My Drive\Ethos LLM\Project_Root"

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
            # Only get headers (first row)
            headers = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
            columns_by_sheet[sheet.title] = [str(h) for h in headers if h]
            if headers:
                text.append("Columns: " + " | ".join([str(h) for h in headers if h]))
            # Do NOT read sample rows for speed!
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
        # Define cache path
        cache_dir = os.path.join(os.path.dirname(path), "_ocr_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")

        # If cache exists, use it
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        # Otherwise, extract text and cache it
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

        # Fallback to OCR if no text extracted
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

def get_relevant_chunks(query, all_chunks, top_k=4):
    query_words = query.lower().split()
    scored = []
    for meta, chunk in all_chunks:
        category_words = []
        if "category" in meta and meta["category"]:
            if isinstance(meta["category"], list):
                for cat in meta["category"]:
                    category_words.extend([w.strip().lower() for w in str(cat).replace(",", " ").split() if w.strip()])
            else:
                category_words = [w.strip().lower() for w in str(meta["category"]).replace(",", " ").split() if w.strip()]
        tag_words = []
        if "tags" in meta and meta["tags"]:
            tag_words = [w.strip().lower() for w in meta["tags"] if w.strip()]
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

def excel_qa(file_path, user_query):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re

    df = pd.read_excel(file_path)
    prompt = (
        f"You are a data analyst working with the following Excel file.\n"
        f"Columns: {list(df.columns)}\n"
        f"Sample data:\n{df.head(5).to_string(index=False)}\n\n"
        f"User question: {user_query}\n\n"
        "Return only valid Python code that uses the provided 'df' DataFrame (do NOT reload or create new data). "
        "If a chart is required, generate and show it using matplotlib/seaborn. "
        "Assign any tabular result to a variable named 'result'.\n"
        "After the code block, provide a brief summary of what the chart or table shows. "
        "Then, in a separate paragraph, suggest at least two possible causes or business insights that could explain the observed pattern, even if you have to speculate based on typical business scenarios."
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

    # Extract explanation text after the code block
    explanation = ""
    if code_match:
        explanation = answer[code_match.end():].strip()

    local_vars = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns}
    plt.clf()

    try:
        exec(code, {}, local_vars)
        if "result" in local_vars:
            st.write("✅ Result:")
            st.dataframe(local_vars["result"])
        st.pyplot(plt)
    except Exception as e:
        st.warning("⚠️ GPT-generated code did not execute successfully.")
        st.text(code)
        st.error(str(e))

    # Show the explanation text if present
    if explanation:
        st.markdown(f"**Explanation:** {explanation}")

# --- Gather all chunks from all files ---
all_files = find_all_files(PROJECT_ROOT)
all_chunks = []
for file_path in all_files:
    ext = os.path.splitext(file_path)[1].lower()
    max_ocr_pages = 5 if ext == ".pdf" else 0
    if ext == ".xlsx":
        text, sheet_names, columns_by_sheet = extract_text_for_metadata(file_path, max_ocr_pages=max_ocr_pages)
    else:
        text = extract_text_for_metadata(file_path, max_ocr_pages=max_ocr_pages)
        sheet_names, columns_by_sheet = [], {}
    if text.strip():
        meta = {"source_file": os.path.basename(file_path)}
        if ext == ".xlsx":
            meta["sheet_names"] = sheet_names
            meta["columns_by_sheet"] = columns_by_sheet
            # Optionally, add a flat list of all columns for easier searching
            all_columns = []
            for cols in columns_by_sheet.values():
                all_columns.extend(cols)
            meta["columns"] = list(set(all_columns))
        # Load or generate metadata
        meta_path = os.path.join(os.path.dirname(file_path), "_metadata", os.path.splitext(os.path.basename(file_path))[0] + ".json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as jf:
                    meta.update(json.load(jf))
            except Exception:
                pass
        else:
            # Generate metadata if not present
            try:
                metadata = auto_generate_metadata(text)
                meta.update(metadata)
                # Save for future use
                meta_dir = os.path.join(os.path.dirname(file_path), "_metadata")
                os.makedirs(meta_dir, exist_ok=True)
                with open(meta_path, "w") as jf:
                    json.dump(metadata, jf, indent=2)
            except Exception:
                pass
        for chunk in chunk_text(text):
            all_chunks.append((meta, chunk))

# (No st.write for total chunks or sample chunk here)

# --- Q&A Section ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_excel" not in st.session_state:
    st.session_state.last_excel = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None

st.header("Ask a question about your knowledge base")

user_query = st.text_input("Your question:", value=st.session_state.get("last_query", ""))
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
        # Normalize category and tags
        category_words = []
        if "category" in meta and meta["category"]:
            if isinstance(meta["category"], list):
                for cat in meta["category"]:
                    category_words.extend([w.strip().lower() for w in str(cat).replace(",", " ").split() if w.strip()])
            else:
                category_words = [w.strip().lower() for w in str(meta["category"]).replace(",", " ").split() if w.strip()]
        tag_words = []
        if "tags" in meta and meta["tags"]:
            tag_words = [w.strip().lower() for w in meta["tags"] if w.strip()]
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

selected_excel = None
if matching_excels:
    best_idx = int(np.argmax(excel_scores))
    selected_excel = st.selectbox(
        "Select an Excel file to answer your question:",
        matching_excels,
        index=best_idx,
        key="excel_select"
    )
    st.info(f"Matching Excel files: {[os.path.basename(f) for f in matching_excels]}")

    # Only run excel_qa if the file or question changed, or button pressed
    run_excel_qa = (
        st.button("Ask") or
        selected_excel != st.session_state.get("last_excel") or
        user_query != st.session_state.get("last_query")
    )
    if run_excel_qa and selected_excel and user_query:
        excel_qa(selected_excel, user_query)
        st.session_state.last_excel = selected_excel
        st.session_state.last_query = user_query
        # Only append to chat history if this is a new question
        if st.session_state.chat_history == [] or st.session_state.chat_history[-1]["content"] != user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": "Chart or result generated above."})
else:
    relevant = get_relevant_chunks(user_query, all_chunks, top_k=4)
    context = "\n\n".join(chunk for meta, chunk in relevant)
    system_prompt = (
        "You are a helpful assistant answering questions based on internal business documents. "
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
        for meta, chunk in relevant:
            st.write(f"**Source:** {meta.get('title', meta.get('source_file', 'N/A'))}")
            st.write(f"**Category:** {meta.get('category', '')}")
            st.write(f"**Tags:** {', '.join(meta.get('tags', []))}")
            st.write(f"**Summary:** {meta.get('summary', '')}")
            st.write(chunk[:500] + ("..." if len(chunk) > 500 else ""))
            st.write("---")

# Place this where you want the chat to appear (after processing the question)
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
    # Show most recent messages at the top
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





