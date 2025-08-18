# logger.py

import os, json, time, io
from pathlib import Path
from constants import META_DIR, LOCAL_META_DIR
try:
    from dbx_utils import append_jsonl_line as _dbx_append
except Exception:
    _dbx_append = None

def _write_local(relname: str, rec: dict) -> str:
    Path(LOCAL_META_DIR).mkdir(parents=True, exist_ok=True)
    lp = f"{LOCAL_META_DIR}/{relname}"
    with open(lp, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return lp

def log_event(message, log_path=f"{META_DIR}/app_events.jsonl"):
    rec = {"ts": time.time(), "message": message}
    lp = _write_local("app_events.jsonl", rec)
    # mirror to Dropbox if the target is a logical path and client exists
    if _dbx_append and (log_path.startswith("/Project_Root") or log_path.startswith("/Apps")):
        try: _dbx_append(log_path, rec)
        except Exception: pass
    return lp

def log_query_result(query, result, save_path=f"{META_DIR}/query_results.jsonl"):
    rec = {"ts": time.time(), "query": query, "result": result}
    lp = _write_local("query_results.jsonl", rec)
    if _dbx_append and (save_path.startswith("/Project_Root") or save_path.startswith("/Apps")):
        try: _dbx_append(save_path, rec)
        except Exception: pass
    return lp