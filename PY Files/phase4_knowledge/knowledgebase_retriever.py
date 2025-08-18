# PY Files/phase4_knowledge/knowledgebase_retriever.py
# Phase 4.2 — Knowledge Base Retriever (top-k + context pack)
from __future__ import annotations

import os, json, argparse, pickle, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

try:
    import tiktoken
except Exception:
    tiktoken = None

# OpenAI client
from openai import OpenAI

# Enterprise foundation imports
try:
    from constants import PROJECT_ROOT
    from path_utils import canon_path
except ImportError:
    # Fallback for standalone usage
    PROJECT_ROOT = "/Project_Root"
    def canon_path(path): return str(Path(path).resolve())

# ---- Config (keep in sync with builder) ----
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
KB_SUBDIR = "06_LLM_Knowledge_Base"
INDEX_REL = f"{KB_SUBDIR}/document_index.faiss"
DOCSTORE_REL = f"{KB_SUBDIR}/docstore.pkl"

# ---- Key loader (same order as builder) ----
def _load_openai_key() -> str:
    try:
        import streamlit as st  # only if running inside Streamlit
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in Streamlit secrets or environment.")
    return key

def _get_client() -> OpenAI:
    return OpenAI(api_key=_load_openai_key())

# ---- Token utils ----
def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    # fallback ~4 chars / token
    return max(1, int(len(text) * 0.25))

# ---- Loaders ----
def load_index_and_docstore(project_root: str) -> Tuple[faiss.IndexFlatIP, Dict[str, list]]:
    root = Path(project_root).resolve()
    idx_path = root / INDEX_REL
    ds_path = root / DOCSTORE_REL
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {idx_path}")
    if not ds_path.exists():
        raise FileNotFoundError(f"Docstore not found at {ds_path}")
    index = faiss.read_index(str(idx_path))
    with open(ds_path, "rb") as f:
        docstore = pickle.load(f)
    return index, docstore

# ---- Embedding ----
def embed_query(query: str, client: Optional[OpenAI] = None, model: str = EMBED_MODEL) -> np.ndarray:
    client = client or _get_client()
    vec = client.embeddings.create(model=model, input=[query]).data[0].embedding
    v = np.array([vec], dtype="float32")
    # cosine via IP requires L2 normalization
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

# ---- Search ----
@dataclass
class SearchHit:
    score: float
    text: str
    meta: Dict[str, Any]

def _passes_filters(meta: Dict[str, Any], source_filter: Optional[Dict[str, List[str]]]) -> bool:
    if not source_filter:
        return True
    # Example: {"source_type": ["pdf","docx"]}
    for key, allowed in source_filter.items():
        val = meta.get(key)
        if val is None:
            return False
        if isinstance(allowed, (list, tuple, set)):
            if val not in allowed:
                return False
        else:
            if val != allowed:
                return False
    return True

def search_topk(
    project_root: str,
    query: str,
    k: int = 5,
    score_threshold: Optional[float] = None,
    source_filter: Optional[Dict[str, List[str]]] = None,
    dedupe_by_doc: bool = True,
    client: Optional[OpenAI] = None,
) -> List[SearchHit]:
    index, docstore = load_index_and_docstore(project_root)
    if index.ntotal <= 0:
        return []

    qv = embed_query(query, client=client)
    k = max(1, min(k, index.ntotal))
    D, I = index.search(qv, k)

    hits: List[SearchHit] = []
    used_doc_ids = set()

    for score, idx in zip(D[0], I[0]):
        if not np.isfinite(score):
            continue
        meta = {}
        text = ""
        if idx < len(docstore.get("vectors", [])):
            meta = dict(docstore["vectors"][idx])
        if idx < len(docstore.get("texts", [])):
            text = docstore["texts"][idx]

        if not _passes_filters(meta, source_filter):
            continue

        if dedupe_by_doc:
            doc_id = meta.get("doc_id")
            if doc_id and doc_id in used_doc_ids:
                continue
            if doc_id:
                used_doc_ids.add(doc_id)

        if score_threshold is not None and float(score) < score_threshold:
            continue

        hits.append(SearchHit(score=float(score), text=text, meta=meta))

    return hits

# ---- Context packing ----
def pack_context(
    hits: List[SearchHit],
    max_tokens: int = 1500,
    separator: str = "\n\n---\n\n",
    include_headers: bool = True,
) -> str:
    """
    Build a single context string within a token budget.
    Packs highest-scoring chunks first; stops before exceeding max_tokens.
    """
    parts: List[str] = []
    total_tokens = 0

    for h in hits:
        header = ""
        if include_headers:
            src = h.meta.get("file_path", "")
            pgr = h.meta.get("page_range")
            src_type = h.meta.get("source_type", "")
            page_info = f" pages {pgr[0]}–{pgr[1]}" if pgr else ""
            header = f"[Source: {src} ({src_type}{page_info}) | score={h.score:.3f}]\n"

        block = f"{header}{h.text}".strip()
        t = _estimate_tokens(block)
        sep_toks = _estimate_tokens(separator) if parts else 0

        if total_tokens + sep_toks + t > max_tokens:
            # try trimming the block to fit
            allowed = max_tokens - total_tokens - sep_toks
            if allowed <= 0:
                break
            # crude trim by character ratio to tokens (~4 chars/token)
            approx_chars = int(allowed / 0.25)
            block = block[:max(0, approx_chars)].rstrip()
            t = _estimate_tokens(block)

            if t == 0:
                break

        if parts:
            parts.append(separator)
            total_tokens += sep_toks

        parts.append(block)
        total_tokens += t

        if total_tokens >= max_tokens:
            break

    return "".join(parts)

# ---- CLI (simple) ----
def _parse_args():
    ap = argparse.ArgumentParser(description="KB Retriever")
    ap.add_argument("--project-root", type=str, required=True, help="Path to Project_Root")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("search", help="Search the KB")
    p_s.add_argument("query", type=str, help="Your query")
    p_s.add_argument("--k", type=int, default=5)
    p_s.add_argument("--score-threshold", type=float, default=None)
    p_s.add_argument("--filter-source-type", type=str, default="", help="csv e.g. pdf,docx,txt")
    p_s.add_argument("--no-dedupe", action="store_true", help="do not dedupe by doc_id")
    p_s.add_argument("--max-context-tokens", type=int, default=1200)

    return ap.parse_args()

def main():
    args = _parse_args()
    if args.cmd == "search":
        sf = None
        if args.filter_source_type:
            types = [s.strip().lower() for s in args.filter_source_type.split(",") if s.strip()]
            sf = {"source_type": types} if types else None

        hits = search_topk(
            project_root=args.project_root,
            query=args.query,
            k=args.k,
            score_threshold=args.score_threshold,
            source_filter=sf,
            dedupe_by_doc=not args.no_dedupe,
        )
        print(json.dumps([{"score": h.score, "meta": h.meta, "preview": h.text[:300]} for h in hits], indent=2))
        ctx = pack_context(hits, max_tokens=args.max_context_tokens)
        print("\n======= PACKED CONTEXT =======\n")
        print(ctx)

if __name__ == "__main__":
    main()
