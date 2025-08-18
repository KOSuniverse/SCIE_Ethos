# session.py

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

class SessionState:
    """
    Stores session-level state for query tracking, file usage, and context.
    Uses standardized metadata folder for logs.
    """
    def __init__(self):
        self.query_log = []
        self.active_file = None
        self.context_memory = []
        self.log_path = f"{META_DIR}/query_log.jsonl"

    def add_entry(self, query: str, result: dict):
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "query": query,
            "result": result
        }
        self.query_log.append(entry)
        self.context_memory.append(result.get("output", {}))

    def save_log_to_file(self, save_path: Optional[str] = None):
        """
        Save session log to metadata folder using JSONL format.
        
        Args:
            save_path: Optional custom path. Uses standard metadata folder if None.
        """
        p = save_path or self.log_path
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        
        with open(p, "a", encoding="utf-8") as f:
            for entry in self.query_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Clear logged entries to avoid duplication
        self.query_log = []

    def reset(self):
        self.query_log = []
        self.active_file = None
        self.context_memory = []
