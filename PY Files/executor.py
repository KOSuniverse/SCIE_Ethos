# executor.py

from classifier import classify_user_intent
from matchmaker import match_query_to_file_sheet
from eda import generate_eda_summary

def run_intent_task(query: str, df_dict: dict, metadata_index: dict, client) -> dict:
    """
    Main dispatcher to run a task based on intent.

    Args:
        query (str): User question.
        df_dict (dict): Loaded DataFrames by file/sheet.
        metadata_index (dict): Metadata about files/sheets.
        client: OpenAI client for classification.

    Returns:
        dict: Result with task type and output.
    """
    # --- Step 1: Classify the query ---
    classification = classify_user_intent(query, client)
    intent = classification["intent"]
    reasoning = classification["reasoning"]

    # --- Step 2: Match files and sheets ---
    matches = match_query_to_file_sheet(query, metadata_index)
    if not matches:
        return {
            "intent": intent,
            "reasoning": reasoning,
            "error": "No matching file or sheet found."
        }

    top_match = matches[0]
    matched_file = top_match["filename"]
    matched_sheet = top_match["matched_sheets"][0]["sheet_name"]
    df = df_dict.get((matched_file, matched_sheet))

    if df is None:
        return {
            "intent": intent,
            "reasoning": reasoning,
            "error": f"No data found for file '{matched_file}', sheet '{matched_sheet}'."
        }

    # --- Step 3: Dispatch task based on intent ---
    if intent == "eda":
        output = generate_eda_summary(df)
    else:
        output = {"message": f"No handler yet for intent: '{intent}'"}

    return {
        "intent": intent,
        "reasoning": reasoning,
        "matched_file": matched_file,
        "matched_sheet": matched_sheet,
        "output": output
    }
