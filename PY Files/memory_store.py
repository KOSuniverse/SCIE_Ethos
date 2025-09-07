# PY Files/memory_store.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List

# Where to persist (Dropbox preferred)
MEM_ROOT = os.getenv("MEMORY_ROOT", "/Project_Root/04_Data/04_Metadata/memory")
PATTERNS = f"{MEM_ROOT}/router_patterns.jsonl"      # routing hints learned
SNIPPETS = f"{MEM_ROOT}/qa_snippets.jsonl"          # short Q&A snippets (optional)

def _ensure_dirs():
    # If running with Dropbox utilities, rely on them; else try local
    try:
        os.makedirs(MEM_ROOT, exist_ok=True)
    except Exception:
        pass

def _append_jsonl(path: str, obj: Dict[str, Any]):
    _ensure_dirs()
    try:
        # Prefer Dropbox helper if present
        from dbx_utils import upload_bytes, read_file_bytes
        try:
            buf = read_file_bytes(path).decode("utf-8")
        except Exception:
            buf = ""
        buf += json.dumps(obj, ensure_ascii=False) + "\n"
        upload_bytes(path, buf.encode("utf-8"))
    except Exception:
        # Fallback local
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def learn_router_success(contract: Dict[str, Any], dp_result: Dict[str, Any], conf_min: float = 0.70):
    """Store small, durable hints only when confidence is good."""
    try:
        # Accept either DP confidence schema or router conf
        dp_conf = float((((dp_result or {}).get("confidence") or {}).get("score")) or 0.0)
        rt_conf = float((contract or {}).get("confidence") or 0.0)
        if max(dp_conf, rt_conf) < conf_min:
            return

        params = (contract or {}).get("params") or {}
        entity = (params.get("entity") or "").strip()
        period = (params.get("period") or "").strip()
        hints = (contract or {}).get("files_hint") or []

        # Try to pull the chosen data sources from DP result
        sources = []
        ms = (dp_result or {}).get("method_and_scope") or {}
        for s in (ms.get("data_sources") or []):
            sources.append(str(s))

        record = {
            "ts": int(time.time()),
            "entity": entity,
            "period": period,
            "hints": hints[:5],
            "sources": sources[:5],
            "router_conf": rt_conf,
            "dp_conf": dp_conf,
        }
        _append_jsonl(PATTERNS, record)

        # Optional: store short executive insight as a snippet
        exec_insight = (dp_result or {}).get("executive_insight") or ""
        if exec_insight:
            _append_jsonl(SNIPPETS, {
                "ts": record["ts"],
                "entity": entity, "period": period,
                "snippet": exec_insight[:500], "sources": sources[:5]
            })
    except Exception:
        # Learning must never break the app
        return

def get_router_context(max_items: int = 40) -> str:
    """Return a tiny text block the Router can use as prior patterns."""
    try:
        from dbx_utils import read_file_bytes
        buf = read_file_bytes(PATTERNS).decode("utf-8")
        lines = [json.loads(x) for x in buf.splitlines() if x.strip()]
    except Exception:
        lines = []
        try:
            with open(PATTERNS, "r", encoding="utf-8") as f:
                lines = [json.loads(x) for x in f if x.strip()]
        except Exception:
            return ""

    # Build a compact, model-friendly context
    items = []
    for r in lines[-max_items:]:
        row = []
        if r.get("entity"): row.append(f"entity={r['entity']}")
        if r.get("period"): row.append(f"period={r['period']}")
        if r.get("hints"):  row.append("hints=" + ", ".join(r["hints"]))
        items.append(" | ".join(row))
    return ("Known routing patterns:\n" + "\n".join(f"- {i}" for i in items)) if items else ""


# --- Admin helpers: list and clear -------------------------------------------------
from typing import List

def _read_jsonl(path: str) -> List[dict]:
    rows = []
    try:
        # Prefer Dropbox if available
        from dbx_utils import read_file_bytes
        try:
            buf = read_file_bytes(path).decode("utf-8")
        except Exception:
            buf = ""
    except Exception:
        # Local fallback
        try:
            with open(path, "r", encoding="utf-8") as f:
                buf = f.read()
        except Exception:
            buf = ""

    for line in buf.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows

def list_router_patterns(max_rows: int = 2000) -> List[dict]:
    rows = _read_jsonl(PATTERNS)
    return rows[-max_rows:]

def clear_router_memory() -> bool:
    _ensure_dirs()
    try:
        # Dropbox first
        from dbx_utils import upload_bytes
        upload_bytes(PATTERNS, b"")
        upload_bytes(SNIPPETS, b"")
        return True
    except Exception:
        try:
            # Local fallback
            open(PATTERNS, "w", encoding="utf-8").close()
            open(SNIPPETS, "w", encoding="utf-8").close()
            return True
        except Exception:
            return False
