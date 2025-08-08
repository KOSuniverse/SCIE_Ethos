# orchestrator.py

import json
from llm_client import get_openai_client
from loader import load_master_metadata_index
from executor import run_intent_task

def run_query_pipeline(query: str, df_dict: dict, metadata_path: str) -> dict:
    """
    Orchestrates the full flow: classify → match → execute → return result.

    Args:
        query (str): User's natural language question.
        df_dict (dict): Dictionary of DataFrames keyed by (filename, sheetname).
        metadata_path (str): Path to the master_metadata_index.json file.

    Returns:
        dict: Full structured result.
    """
    # Load OpenAI client
    client = get_openai_client()

    # Load metadata index
    metadata_index = load_master_metadata_index(metadata_path)

    # Run task pipeline
    result = run_intent_task(query, df_dict, metadata_index, client)

    return result
