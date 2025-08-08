# logger.py

import os
import json
from datetime import datetime

def log_event(message: str, log_path: str):
    """
    Appends a timestamped message to a log file.

    Args:
        message (str): Message to log.
        log_path (str): Path to the log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

def log_query_result(query: str, result: dict, save_path: str):
    """
    Saves the full query and result to a structured JSON log.

    Args:
        query (str): User question.
        result (dict): Output from orchestrator.
        save_path (str): JSON file path.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "result": result
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.append(log_entry)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
