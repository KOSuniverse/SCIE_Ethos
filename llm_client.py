# --- llm_client.py ---

import numpy as np
import json
import streamlit as st
from openai import OpenAI
from utils.retry import openai_with_retry

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"


def validate_embedding(vec, expected_dim=1536):
    if vec is None or not isinstance(vec, np.ndarray) or vec.shape[0] != expected_dim:
        return np.zeros(expected_dim)
    return vec


def get_embedding(text):
    try:
        response = openai_with_retry(
            lambda: client.embeddings.create(
                model=EMBEDDING_MODEL,
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


def answer_question(user_query, context):
    prompt = f"""
You are a business analyst. Use the context below to answer the question.

Context:
{context[:4000]}

Question: {user_query}

Answer:
"""
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM answer failed: {e}"


def verify_answer(answer, context, user_query):
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a fact checker. Given a context and an answer, rate the factual accuracy (0-100) and list any unsupported claims."},
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}\n\nQuestion:\n{user_query}"}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Verification failed: {e}"


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
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json") or raw.startswith("```"):
            raw = raw.strip("`").replace("json", "").strip()
        return json.loads(raw)
    except Exception as e:
        st.warning(f"LLM metadata generation failed: {e}")
        return {}


def mine_for_root_causes(user_query, all_chunks, top_chunks):
    mining_prompt = """
You are a senior data analyst trained to identify operational root causes across business documents such as Excel files, reports, audits, or presentations.

Your task is to analyze the provided context in response to a user’s question. Using supporting content from the data, identify possible root causes, key drivers, or anomalies.

Return your analysis in this format (JSON only):
{
  "root_cause_summary": "Concise paragraph explaining the issue",
  "supporting_evidence": [
    {"file": "filename.xlsx", "reason": "Column 'usage' drops in March while 'inventory' spikes"},
    {"file": "audit_report.pdf", "reason": "Notes a stock adjustment not reflected in system"}
  ],
  "confidence_score": 0–100
}
"""
    all_context = "\n\n".join([chunk for _, chunk in all_chunks])
    top_context = "\n\n".join([chunk for _, chunk in top_chunks])
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": mining_prompt},
                {"role": "user", "content": f"Question:\n{user_query}\n\nRelevant context:\n{top_context}\n\nAll data:\n{all_context[:8000]}"}
            ],
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json") or raw.startswith("```"):
            raw = raw.strip("`").replace("json", "").strip()
        return json.loads(raw)
    except Exception:
        return {}

