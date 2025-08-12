# PY Files/assistant_bridge.py
import os, json, time
from openai import OpenAI
from confidence import score_ravc, should_abstain  # moved from app.confidence :contentReference[oaicite:2]{index=2}

def auto_model(intent: str) -> str:
    if intent in {"root_cause", "forecasting", "scenario", "anomaly", "exec_summary"}:
        return "gpt-4o"
    return "gpt-4o-mini"

def _extract_answer(client: OpenAI, thread_id: str) -> str:
    """
    Robustly extract the latest assistant message text.
    Handles multiple content parts and non-text parts gracefully.
    """
    msgs = client.beta.threads.messages.list(thread_id=thread_id, order="asc").data
    assistants = [m for m in msgs if getattr(m, "role", "") == "assistant"]
    if not assistants:
        return ""
    last = assistants[-1]
    parts = getattr(last, "content", []) or []
    # prefer all text parts concatenated
    text_chunks = []
    for p in parts:
        try:
            if getattr(p, "type", "") == "text" and hasattr(p, "text") and getattr(p.text, "value", None):
                text_chunks.append(p.text.value)
        except Exception:
            continue
    if text_chunks:
        return "\n\n".join(text_chunks)
    # fallback to first text-ish thing if structure changes
    try:
        return parts[0].text.value  # original approach :contentReference[oaicite:3]{index=3}
    except Exception:
        return ""

def run_query(question: str, intent_hint: str | None = None) -> dict:
    client = OpenAI()
    with open("config/assistant.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assistant_id = meta["assistant_id"]

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=question)

    # Let the Assistant use File Search + Code Interpreter
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    # Poll until complete (hosted runner does this synchronously)
    terminal_states = {"completed", "failed", "cancelled", "expired"}
    while True:
        r = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if r.status in terminal_states:
            break
        time.sleep(1.2)

    # Extract answer text defensively
    answer = _extract_answer(client, thread.id)

    # Simple confidence example; replace with real signals
    conf = score_ravc(recency=0.8, alignment=0.8, variance=0.2, coverage=0.8)  # :contentReference[oaicite:4]{index=4}
    if should_abstain(conf["score"], threshold=0.52):  # :contentReference[oaicite:5]{index=5}
        answer = "I’m not confident enough to answer. Please refine the question."

    # Log to S3 (best-effort; never block the reply)
    bucket = os.getenv("S3_BUCKET")
    prefix = os.getenv("S3_PREFIX", "project-root")
    if bucket:
        try:
            import boto3, datetime as dt
            s3 = boto3.client("s3")
            log = {
                "ts": dt.datetime.utcnow().isoformat() + "Z",
                "q": question,
                "answer": answer,
                "confidence": conf,
            }
            key = f"{prefix}/logs/query_log.jsonl"
            # append-ish write (racy but fine for low concurrency)
            try:
                old = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            except Exception:
                old = ""
            s3.put_object(Bucket=bucket, Key=key, Body=(old + json.dumps(log) + "\n").encode("utf-8"))
        except Exception:
            # swallow logging errors
            pass

    return {"answer": answer, "confidence": conf}

def assistant_summarize(content_type: str, data: str, context: str = "") -> str:
    """
    Assistant-driven summarization for various content types.
    Used for final EDA summaries and other content processing.
    """
    try:
        prompt = f"""
        Please provide a professional summary of this {content_type}.
        
        Content to summarize:
        {data}
        
        Additional context: {context}
        
        Focus on key insights, trends, and actionable findings.
        Format as a clear, executive-level summary.
        """
        
        result = run_query(prompt)
        return result.get("answer", data)  # Fallback to original data
        
    except Exception as e:
        print(f"Assistant summarization failed: {e}")
        return data  # Fallback to original data

def assistant_eda_plan(schema: dict, sample_data: list, user_question: str = "") -> dict:
    """
    Assistant-driven EDA planning based on data schema and sample.
    Returns suggested analysis steps and visualizations.
    """
    try:
        prompt = f"""
        Based on this data schema and sample, suggest an EDA analysis plan.
        
        Schema: {json.dumps(schema, indent=2)}
        Sample data: {json.dumps(sample_data[:5], indent=2)}
        User question: {user_question}
        
        Return a JSON plan with:
        {{
            "analysis_steps": ["step1", "step2", ...],
            "recommended_charts": [
                {{"type": "bar", "x": "column", "y": "column", "title": "..."}},
                ...
            ],
            "key_metrics": ["metric1", "metric2", ...],
            "insights": ["insight1", "insight2", ...]
        }}
        """
        
        result = run_query(prompt)
        
        # Try to parse JSON from response
        try:
            plan = json.loads(result.get("answer", "{}"))
            return plan
        except json.JSONDecodeError:
            # Extract JSON if wrapped in text
            import re
            answer = result.get("answer", "")
            json_match = re.search(r'\{[\s\S]*\}', answer)
            if json_match:
                return json.loads(json_match.group())
            
        # Fallback plan
        return {
            "analysis_steps": ["Basic statistics", "Distribution analysis", "Correlation analysis"],
            "recommended_charts": [{"type": "bar", "title": "Data Overview"}],
            "key_metrics": ["count", "mean", "sum"],
            "insights": ["Data loaded successfully"]
        }
        
    except Exception as e:
        print(f"Assistant EDA planning failed: {e}")
        return {
            "analysis_steps": ["Basic analysis"],
            "recommended_charts": [],
            "key_metrics": [],
            "insights": [f"Planning failed: {str(e)}"]
        }

def choose_aggregation(user_q: str, schema: dict, sample_rows: list) -> dict:
    """
    NEW: Assistant-driven aggregation plan selection.
    Analyzes user question against data schema to determine optimal aggregation strategy.
    
    Returns JSON plan: {op, col, groupby, filters}
    """
    try:
        # Prepare schema summary
        columns = list(schema.keys()) if isinstance(schema, dict) else schema
        sample_str = json.dumps(sample_rows[:3], indent=2) if sample_rows else "No sample data"
        
        prompt = f"""
        Analyze this user question and data to determine the best aggregation strategy.
        
        User Question: "{user_q}"
        
        Available Columns: {columns}
        
        Sample Data:
        {sample_str}
        
        Return a JSON aggregation plan:
        {{
            "op": "sum|count|mean|max|min|group",
            "col": "primary_column_to_aggregate",
            "groupby": ["column1", "column2"] or null,
            "filters": [
                {{"col": "column", "op": "==|>|<|contains", "value": "filter_value"}}
            ] or null,
            "reasoning": "explanation of chosen approach"
        }}
        
        Guidelines:
        - For "how much" questions, use "sum" with value columns
        - For "how many" questions, use "count" 
        - For "by category" questions, include appropriate groupby
        - Choose the most relevant columns based on the question
        """
        
        result = run_query(prompt)
        answer = result.get("answer", "{}")
        
        try:
            # Parse JSON response
            plan = json.loads(answer)
            
            # Validate plan structure
            if not isinstance(plan, dict):
                raise ValueError("Plan is not a dictionary")
                
            # Ensure required fields
            plan.setdefault("op", "sum")
            plan.setdefault("col", columns[0] if columns else "value")
            plan.setdefault("groupby", None)
            plan.setdefault("filters", None)
            plan.setdefault("reasoning", "Default aggregation plan")
            
            return plan
            
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', answer)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        # Fallback plan based on simple heuristics
        return _create_fallback_aggregation_plan(user_q, columns)
        
    except Exception as e:
        print(f"Assistant aggregation planning failed: {e}")
        return _create_fallback_aggregation_plan(user_q, schema if isinstance(schema, list) else list(schema.keys()))

def _create_fallback_aggregation_plan(user_q: str, columns: list) -> dict:
    """Create a simple aggregation plan when Assistant is unavailable."""
    user_q_lower = user_q.lower()
    
    # Find value columns
    value_cols = [col for col in columns if any(term in str(col).lower() 
                  for term in ['cost', 'value', 'amount', 'price', 'wip', 'total'])]
    
    # Find grouping columns
    group_cols = [col for col in columns if any(term in str(col).lower() 
                  for term in ['type', 'category', 'location', 'department', 'part'])]
    
    # Determine operation
    if any(term in user_q_lower for term in ['how much', 'total', 'sum']):
        op = "sum"
        col = value_cols[0] if value_cols else (columns[0] if columns else "value")
    elif any(term in user_q_lower for term in ['how many', 'count']):
        op = "count"
        col = columns[0] if columns else "id"
    elif any(term in user_q_lower for term in ['average', 'mean']):
        op = "mean"
        col = value_cols[0] if value_cols else (columns[0] if columns else "value")
    else:
        op = "sum"
        col = value_cols[0] if value_cols else (columns[0] if columns else "value")
    
    # Determine grouping
    groupby = None
    if any(term in user_q_lower for term in ['by', 'per', 'each', 'breakdown']):
        groupby = group_cols[:2] if group_cols else None
    
    return {
        "op": op,
        "col": col,
        "groupby": groupby,
        "filters": None,
        "reasoning": f"Fallback plan: {op} on {col}" + (f" grouped by {groupby}" if groupby else "")
    }

