from datetime import datetime
from s3_utils import list_objects, upload_json, download_json, append_jsonl_line

print("Listing bucket…", list_objects(prefix=""))  # should run without error

# write/read JSON
path = upload_json({"hello": "world", "ts": datetime.utcnow().isoformat()}, "config/_smoke.json", indent=2)
print("Uploaded:", path)
print("Downloaded:", download_json("config/_smoke.json"))

# append JSONL log line
append_jsonl_line("logs/query_log.jsonl", {"event": "smoke", "ts": datetime.utcnow().isoformat()})
print("Appended to logs/query_log.jsonl")
