# llm_client.py

import os
import openai
from dotenv import load_dotenv

# Optional import if using Streamlit UI
try:
    import streamlit as st
    streamlit_available = True
except ImportError:
    streamlit_available = False

# --- Load environment variables ---
load_dotenv()

def get_openai_api_key():
    """
    Retrieves the OpenAI API key from .env or Streamlit secrets.
    """
    # Priority 1: .env file
    key = os.getenv("OPENAI_API_KEY")

    # Priority 2: Streamlit secrets
    if not key and streamlit_available and hasattr(st, "secrets"):
        key = st.secrets.get("OPENAI_API_KEY", None)

    # If still missing
    if not key:
        raise ValueError("❌ OpenAI API key not found in .env or Streamlit secrets.")
    
    return key

def get_openai_client():
    """
    Returns an OpenAI client using the resolved API key.
    """
    api_key = get_openai_api_key()
    return openai.OpenAI(api_key=api_key)
