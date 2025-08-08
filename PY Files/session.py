# session.py

import os
import json
from datetime import datetime

class SessionState:
    """
    Stores session-level state for query tracking, file usage, and context.
    """
    def __init__(self):
        self.query_log = []
        self.active_file = None
        self.context_memory = []

    def add_entry(self, query: str, result: dict):
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "query": query,
            "result": result
        }
        self.query_log.append(entry)
        self.context_memory.append(result.get("output", {}))

    def save_log_to_file(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.query_log, f, indent=2)

    def reset(self):
        self.query_log = []
        self.active_file = None
        self.context_memory = []
