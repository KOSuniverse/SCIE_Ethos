# --- excel_qa.py ---

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from utils.gdrive import download_file
from llm_client import answer_question
from utils.metadata import load_learned_answers, save_learned_answers


def excel_qa(file_path, user_query, column_aliases=None):
    try:
        file_stream = download_file(file_path)
        df = pd.read_excel(file_stream)

        learned_answers = load_learned_answers()
        if user_query in learned_answers:
            cached = learned_answers[user_query]
            st.success("✅ Reused cached answer for this query")
            st.markdown("**Answer:**")
            st.markdown(cached.get("answer", ""))
            if cached.get("chart_image"):
                st.image(cached["chart_image"], caption="Cached Chart")
            return

        auto_chart = any(word in user_query.lower() for word in [
            "trend", "compare", "distribution", "growth", "pattern", "chart", "plot", "visual"])

        prompt = (
            f"You are a data analyst working with Excel data.\n"
            f"The user asked: '{user_query}'\n\n"
            f"Here are column aliases (user-defined):\n{column_aliases}\n\n"
            f"Columns in the file: {list(df.columns)}\n"
            f"First few rows:\n{df.head(5).to_string(index=False)}\n\n"
            "Instructions:\n"
            "1. Use the 'df' DataFrame directly (don't reload data).\n"
            "2. Summarize or analyze data to answer the question.\n"
            f"3. {'Create a matplotlib/seaborn chart if helpful.' if auto_chart else 'Chart is optional.'}\n"
            "4. Return valid Python code that defines 'result' and generates any chart.\n"
            "5. After the code block, explain the output and any business insight."
        )

        response = answer_question(user_query, prompt)

        code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else response.strip()
        explanation = response[code_match.end():].strip() if code_match else ""

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

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            chart_image_bytes = buffer.read()

            # Save learned answer
            learned_answers[user_query] = {
                "answer": explanation,
                "chart_image": chart_image_bytes.hex()
            }
            save_learned_answers(learned_answers)

        except Exception as e:
            st.warning("⚠️ GPT-generated code did not execute successfully.")
            st.text(code)
            st.error(str(e))

        if explanation:
            st.markdown("**Explanation:**")
            st.markdown(explanation)

    except Exception as e:
        st.error(f"Excel Q&A failed: {e}")


