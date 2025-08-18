import streamlit as st
import sys, platform
import numpy as np, pandas as pd
import _snowflake   # 👈 add this

st.set_page_config(page_title="Smoke Test", layout="wide")

st.success("✅ Streamlit in Snowflake is running.")
st.write(
    {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
)

# 👇 NEW: check secrets
assistant_id = _snowflake.get_generic_secret_string("ASSISTANT_ID")
openai_key   = _snowflake.get_generic_secret_string("OPENAI_API_KEY")

st.write("✅ Secrets wired up.")
st.write("Assistant ID starts with:", assistant_id[:8])  # safe preview
st.write("OpenAI key starts with:", openai_key[:5])

