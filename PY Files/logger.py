# logger.py

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import constants
try:
    from .constants import META_DIR
except ImportError:
    # Fallback for standalone usage
    from constants import META_DIR

def log_event(message: str, log_path: Optional[str] = None):
    """
    Appends a timestamped message to a log file using JSONL format.
    Uses standardized metadata folder if no path specified.

    Args:
        message (str): Message to log.
        log_path (str): Optional custom path. Uses metadata folder if None.
    """
    if log_path is None:
        log_path = f"{META_DIR}/app_events.jsonl"
    
    rec = {"ts": time.time(), "message": message}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def log_query_result(query: str, result: dict, save_path: Optional[str] = None):
    """
    Saves the full query and result to a structured JSONL log.
    Uses standardized metadata folder if no path specified.

    Args:
        query (str): User question.
        result (dict): Output from orchestrator.
        save_path (str): Optional custom path. Uses metadata folder if None.
    """
    if save_path is None:
        save_path = f"{META_DIR}/query_results.jsonl"
    
    rec = {"ts": time.time(), "query": query, "result": result}
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
