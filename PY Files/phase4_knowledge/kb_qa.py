# PY Files/phase4_knowledge/kb_qa.py
# Phase 4.4 — Full KB QA pipeline
# search -> pack_context -> compose_response

from __future__ import annotations
from typing import Optional, Dict, Any

from phase4_knowledge.knowledgebase_retriever import search_topk, pack_context
from phase4_knowledge.response_composer import compose_response

def kb_answer(
    project_root: str,
    query: str,
    k: int = 5,
    score_threshold: Optional[float] = None,
    source_filter: Optional[Dict[str, Any]] = None,
    dedupe_by_doc: bool = True,
    max_context_tokens: int = 1500,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_answer_tokens: int = 800,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search the KB, pack context, and get an LLM answer.
    Returns dict with hits, context, and answer.
    """
    hits = search_topk(
        project_root=project_root,
        query=query,
        k=k,
        score_threshold=score_threshold,
        source_filter=source_filter,
        dedupe_by_doc=dedupe_by_doc,
    )

    context = pack_context(hits, max_tokens=max_context_tokens)
    answer = compose_response(
        query=query,
        context=context,
        model=model,
        temperature=temperature,
        max_tokens=max_answer_tokens,
        system_prompt=system_prompt,
    )

    return {
        "query": query,
        "hits": hits,
        "context": context,
        "answer": answer
    }

# CLI for quick testing
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="KB QA Pipeline")
    ap.add_argument("--project-root", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    res = kb_answer(args.project_root, args.query, k=args.k)
    print(json.dumps({
        "query": res["query"],
        "answer": res["answer"],
        "context_preview": res["context"][:500]
    }, indent=2))
