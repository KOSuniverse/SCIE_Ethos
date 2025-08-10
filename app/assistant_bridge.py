# app/assistant_bridge.py
import os, json, time
from openai import OpenAI
from app.confidence import score_ravc, should_abstain

def auto_model(intent: str) -> str:
    if intent in {"root_cause","forecasting","scenario","anomaly","exec_summary"}:
        return "gpt-4o"
    return "gpt-4o-mini"

def run_query(question: str, intent_hint: str | None = None) -> dict:
    client = OpenAI()
    with open("config/assistant.json","r",encoding="utf-8") as f:
        meta = json.load(f)
    assistant_id = meta["assistant_id"]

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=question)

    # Let the Assistant use File Search + Code Interpreter
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    # Poll until complete (your hosted runner does this synchronously)
    while True:
        r = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if r.status in {"completed","failed","cancelled","expired"}:
            break
        time.sleep(1.2)

    messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc").data
    answer = [m for m in messages if m.role=="assistant"][-1].content[0].text.value

    # Simple confidence example; replace with real signals
    conf = score_ravc(recency=0.8, alignment=0.8, variance=0.2, coverage=0.8)
    if should_abstain(conf["score"], threshold=0.52):
        answer = "I’m not confident enough to answer. Please refine the question."

    # Log to S3 (if configured)
    bucket = os.getenv("S3_BUCKET"); prefix = os.getenv("S3_PREFIX","project-root")
    if bucket:
        import boto3, datetime as dt
        s3 = boto3.client("s3")
        log = {"ts": dt.datetime.utcnow().isoformat()+"Z", "q": question, "answer": answer, "confidence": conf}
        key = f"{prefix}/logs/query_log.jsonl"
        # append-ish write (best-effort)
        try:
            old = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        except s3.exceptions.NoSuchKey:
            old = ""
        s3.put_object(Bucket=bucket, Key=key, Body=(old + json.dumps(log)+"\n").encode("utf-8"))

    return {"answer": answer, "confidence": conf}
