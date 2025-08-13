# PY Files/orchestrator.py
"""
SIMPLE WORKING ORCHESTRATOR - Restored to working local temp version
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Simple imports - no complex enterprise stuff
from constants import LOCAL_META_DIR
from file_utils import list_cleaned_files
from loader import load_excel_file

# Core imports with fallbacks
try:
    from llm_client import get_openai_client
except Exception:
    def get_openai_client():
        raise RuntimeError("llm_client unavailable")

try:
    from intent import classify_intent
except Exception:
    def classify_intent(question):
        return {"intent": "eda", "confidence": "0.50", "reason": "Intent classifier unavailable"}

try:
    from executor import run_intent_task
except Exception:
    def run_intent_task(*args, **kwargs):
        return {"error": "Executor unavailable", "text": "Executor module could not be imported"}


# SIMPLE WORKING FUNCTION - This was giving you good answers
def answer_question(
    user_question: str,
    *,
    app_paths: Any,
    cleansed_paths: Optional[List[str]] = None,
    answer_style: str = "detailed",
) -> Dict[str, Any]:
    """
    SIMPLE APPROACH: Just do the basic processing that was working
    """
    print(f"🎯 SIMPLE: Processing question: {user_question}")
    
    try:
        # 1. Intent classification  
        intent_info = classify_intent(user_question)
        print(f"🎯 SIMPLE: Intent={intent_info.get('intent')}, Confidence={intent_info.get('confidence')}")
        
        # 2. Load files
        df_dict = {}
        files = list_cleaned_files()
        print(f"🎯 SIMPLE: Found {len(files)} files")
        
        for p in files:
            try:
                df_dict[p] = load_excel_file(p, "xlsx") 
                print(f"🎯 SIMPLE: Loaded {p}")
            except Exception as e:
                print(f"🎯 SIMPLE: Failed to load {p}: {e}")
        
        # 3. Get OpenAI client
        try:
            client = get_openai_client()
        except Exception:
            client = None
            
        # 4. Run executor
        res = run_intent_task(user_question, df_dict, {}, client)
        
        # 5. Format result for main.py
        return {
            "text": res.get("text", "No answer generated"),
            "final_text": res.get("text", "No answer generated"), 
            "intent_info": intent_info,
            "calls": [{"tool": "executor", "result": res}],
            "context": {"last_rows": []},
            "artifacts": res.get("artifacts", []),
            "tool_calls": [{"tool": "executor", "result": res}]
        }
        
    except Exception as e:
        print(f"🎯 SIMPLE: Error: {e}")
        return {
            "text": f"Error: {str(e)}",
            "final_text": f"Error: {str(e)}",
            "intent_info": {"intent": "error", "confidence": "0.00", "reason": str(e)},
            "calls": [],
            "context": {"last_rows": []},
            "artifacts": [],
            "tool_calls": []
        }


def answer_question_simple(user_question: str) -> str:
    """
    Even simpler wrapper for testing
    """
    print(f"🎯 SIMPLE: Processing question: {user_question}")
    
    try:
        # 1. Intent classification  
        intent_info = classify_intent(user_question)
        print(f"🧠 SIMPLE: Intent={intent_info.get('intent')}, Confidence={intent_info.get('confidence')}")
        
        # 2. Load files
        df_dict = {}
        files = list_cleaned_files()
        print(f"📊 SIMPLE: Found {len(files)} files")
        
        for p in files:
            try:
                df_dict[p] = load_excel_file(p, "xlsx")
                print(f"📊 SIMPLE: Loaded {p}")
            except Exception as e:
                print(f"⚠️ SIMPLE: Failed to load {p}: {e}")
        
        # 3. Get OpenAI client
        try:
            client = get_openai_client()
        except Exception:
            client = None
            
        # 4. Run executor
        res = run_intent_task(user_question, df_dict, {}, client)
        
        # Add intent info to response
        if isinstance(res, dict):
            res["intent_info"] = intent_info
            
        return json.dumps(res, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ SIMPLE: Error: {e}")
        error_response = {
            "error": str(e),
            "text": f"Sorry, I encountered an error: {str(e)}",
            "intent_info": {"intent": "error", "confidence": "0.00", "reason": str(e)}
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)


# Minimal compatibility functions for legacy imports
def run_ingest_pipeline(source: bytes | bytearray | str, filename: Optional[str], paths: Optional[Any]) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """Placeholder for ingest pipeline"""
    return {}, {"status": "ingest_not_implemented"}

