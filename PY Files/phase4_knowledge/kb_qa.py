# PY Files/phase4_knowledge/kb_qa.py
# Phase 4.4 — Full KB QA pipeline with Assistant integration
# search -> pack_context -> compose_response (with optional Assistant enhancement)

from __future__ import annotations
from typing import Optional, Dict, Any

from phase4_knowledge.knowledgebase_retriever import search_topk, pack_context
from phase4_knowledge.response_composer import compose_response

# Optional Assistant integration for enhanced KB answers
try:
    from assistant_bridge import run_query as assistant_run_query
    ASSISTANT_AVAILABLE = True
except ImportError:
    ASSISTANT_AVAILABLE = False

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
    use_assistant: bool = True,  # NEW: Option to use Assistant for enhanced answers
) -> Dict[str, Any]:
    """
    Search the KB, pack context, and get an LLM answer.
    
    Enhanced with optional Assistant integration for supply chain expertise.
    
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
    
    # Enhanced: Try Assistant first for supply chain domain expertise
    answer = ""
    answer_source = "standard"
    
    if use_assistant and ASSISTANT_AVAILABLE and context.strip():
        try:
            # Use Assistant for enhanced KB answers with supply chain context
            assistant_prompt = f"""
            Based on the following supply chain and inventory management knowledge base content, 
            please provide a comprehensive answer to the user's question.
            
            Knowledge Base Context:
            {context}
            
            User Question: {query}
            
            Please provide a detailed, actionable answer based on the knowledge base content. 
            Focus on practical supply chain insights, best practices, and specific recommendations.
            If the knowledge base doesn't contain sufficient information, clearly state this.
            """
            
            assistant_result = assistant_run_query(assistant_prompt)
            if isinstance(assistant_result, dict) and assistant_result.get("answer"):
                answer = assistant_result["answer"]
                answer_source = "assistant_enhanced"
            elif isinstance(assistant_result, str):
                answer = assistant_result
                answer_source = "assistant_enhanced"
        except Exception as e:
            # Fall back to standard response composer
            print(f"Assistant KB enhancement failed: {e}")
    
    # Fall back to standard response composer if Assistant not available or failed
    if not answer.strip():
        answer = compose_response(
            query=query,
            context=context,
            model=model,
            temperature=temperature,
            max_tokens=max_answer_tokens,
            system_prompt=system_prompt,
        )
        answer_source = "standard"

    return {
        "query": query,
        "hits": hits,
        "context": context,
        "answer": answer,
        "answer_source": answer_source,  # Track which method was used
        "assistant_available": ASSISTANT_AVAILABLE
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
