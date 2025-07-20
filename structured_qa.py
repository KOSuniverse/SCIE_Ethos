# --- structured_qa.py ---

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from utils.gdrive import download_file
from llm_client import answer_question


def structured_data_qa(user_query, top_chunks):
    excel_dfs = []
    doc_texts = []

    # Separate structured Excel and unstructured content
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

    # Prepare Excel prompt samples
    excel_prompt = "\n\n".join([
        f"File: {f}, Sheet: {s}\n(df = this sheet)\n{d.head(5).to_string(index=False)}"
        for f, s, d in excel_dfs
    ])
    doc_context = "\n\n".join(doc_texts)

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

🧠 Instructions:
- Be specific — cite values, fields, sheets, or observations.
- If relevant, generate a matplotlib/seaborn chart using the `df` variable.
- Return only markdown-formatted content.
- If including code, wrap it in a Python code block.
- Your answer should be clear to business and data users.

Return your full response in Markdown format.
"""

    try:
        response = answer_question(user_query, prompt)
        code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None
        explanation = response[code_match.end():].strip() if code_match else response

        st.markdown("### 🤖 GPT Analysis")
        st.markdown(explanation)

        if code:
            plt.clf()
            local_vars = {"pd": pd, "plt": plt}
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
                st.text(code)
                st.text(str(e))

    except Exception as e:
        st.error(f"Structured Q&A failed: {e}")


# --- OPTIONAL BUTTON TO RUN FROM main.py ---

def show_structured_qa_button(user_query, top_chunks):
    st.markdown("---")
    if st.button("📊 Run Structured Excel + Document Analysis"):
        structured_data_qa(user_query, top_chunks)


# --- Prompt for model save name in modeling.py ---
def model_name_input():
    return st.text_input("Save model as filename (include .pkl):", value="auto_model.pkl")
