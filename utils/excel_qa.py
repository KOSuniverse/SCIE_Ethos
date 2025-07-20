# --- utils/excel_qa.py ---

import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from openai import OpenAI
from utils.gdrive import download_file

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
CHAT_MODEL = "gpt-4o"

def excel_qa(file_id, user_query, column_aliases=None):
    """
    Advanced Excel Q&A with automatic chart generation and business insights.
    
    Args:
        file_id: Google Drive file ID for the Excel file
        user_query: User's question about the data
        column_aliases: Column alias mappings for this file
    """
    try:
        # Download and load Excel file
        file_stream = download_file(file_id)
        df = pd.read_excel(file_stream)
        
        # Detect if user wants visualization
        auto_chart = any(
            word in user_query.lower() 
            for word in ["trend", "compare", "distribution", "growth", "pattern", "chart", "plot", "visual"]
        )
        
        # Build comprehensive prompt
        prompt = (
            f"Column aliases for this file: {json.dumps(column_aliases or {})}\n"
            f"You are a data analyst working with the following Excel file.\n"
            f"Columns: {list(df.columns)}\n"
            f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
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
            "Include plt.title() for any charts you create.\n"
            "After the code block, provide your explanation and insights."
        )
        
        # Get LLM response
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract code block
        code_match = re.search(r"```(?:python)?(.*?)```", answer, re.DOTALL)
        code = code_match.group(1).strip() if code_match else answer.strip()
        
        # Extract explanation
        explanation = ""
        if code_match:
            explanation = answer[code_match.end():].strip()
        
        # Set up execution environment
        local_vars = {
            "df": df.copy(), 
            "pd": pd, 
            "plt": plt, 
            "sns": sns
        }
        
        # Clear any existing plots
        plt.clf()
        
        # Execute the generated code
        try:
            exec(code, {}, local_vars)
            
            # Display tabular results if available
            if "result" in local_vars:
                st.write("✅ Analysis Result:")
                result = local_vars["result"]
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result.head(20))  # Limit display for large results
                else:
                    st.write(result)
            
            # Display chart if created
            if plt.get_fignums():  # Check if any figures were created
                st.pyplot(plt)
            
        except Exception as e:
            st.warning("⚠️ GPT-generated code did not execute successfully.")
            st.code(code, language="python")
            st.error(str(e))
        
        # Display explanation and insights
        if explanation:
            st.markdown("**📊 Analysis & Insights:**")
            st.markdown(explanation)
        
    except Exception as e:
        st.error(f"Excel Q&A failed: {e}")

def structured_data_qa(user_query, top_chunks):
    """
    Perform dual analysis of structured Excel data and unstructured document content.
    
    Args:
        user_query: User's question
        top_chunks: Top-ranked chunks from similarity search
    """
    excel_dfs = []
    doc_texts = []
    
    # Separate structured Excel data and unstructured document chunks
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
    
    # Prepare structured Excel sample prompt
    excel_context = []
    for file, sheet, df in excel_dfs:
        context = f"File: {file}, Sheet: {sheet}\n(df = this sheet)\n{df.head(5).to_string(index=False)}"
        excel_context.append(context)
    
    excel_prompt = "\n\n".join(excel_context)
    doc_context = "\n\n".join(doc_texts)
    
    # Build comprehensive GPT prompt
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
    
    # Call GPT to analyze both structured and unstructured content
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior data analyst responding in markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        reply = response.choices[0].message.content.strip()
        
        # Extract code block if any, and explanation
        code_match = re.search(r"```(?:python)?(.*?)```", reply, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None
        explanation = reply[code_match.end():].strip() if code_match else reply
        
        # Display insights first
        st.markdown("### 🤖 Comprehensive Analysis")
        st.markdown(explanation)
        
        # Execute GPT-generated charting code and display/download chart
        if code:
            plt.clf()
            local_vars = {"pd": pd, "plt": plt, "sns": sns}
            
            # Add DataFrames to local vars for code execution
            for i, (file, sheet, df) in enumerate(excel_dfs):
                local_vars[f"df_{i}"] = df
                if i == 0:  # Make first DataFrame available as 'df'
                    local_vars["df"] = df
            
            try:
                exec(code, {}, local_vars)
                
                # Show chart in Streamlit
                if plt.get_fignums():
                    st.pyplot(plt)
                
                # Show result table if exists
                if "result" in local_vars:
                    st.dataframe(local_vars["result"])
                    
            except Exception as e:
                st.error("⚠️ Error running GPT-generated chart code")
                st.code(code, language="python")
                st.text(str(e))
    
    except Exception as e:
        st.error(f"Structured Q&A failed: {e}")
