# Merge Guide – SCIE Ethos (Cursor Build)

Detected repo layout: root contains `main.py` and **no** `/prompts` folder.

## What to do
1) Unzip this archive at the **root** of your repository (same level as `main.py`). It will add:
   - `/prompts` (instructions + aliases + modeling + ingest rules)
   - `/schemas` (catalog + turn-log schemas)
   - `/templates` (summary card template)
   - `/configs` (orchestrator rules, export + retention policies)
   - `/qa` (acceptance tests)
   - `CURSOR_PROMPT.md` and `UI_SPEC_ADDENDUM.md`

2) Open the repo in Cursor and paste the contents of **CURSOR_PROMPT.md** into the chat.
   - Cursor will first **audit & refactor** the existing repo to align with this layout
     (it will keep your `main.py` and integrate the new folders).
   - It will then create the Assistant, implement Dropbox→Assistant sync per `prompts/ingest_rules.yaml`,
     wire the orchestrator, and run the QA suite.

3) Secrets to verify (Streamlit Cloud project settings or `secrets.toml`):
   - `OPENAI_API_KEY`, Dropbox app creds, AWS S3 creds, region, and bucket/prefix values.

4) Smoke tests to try:
   - “Show US inventory by value and aging—real insights.”
   - “Par today and by Q2-2026 for SKU 12345 in Germany—include SL sensitivity.”

## Notes
- We intentionally **did not** overwrite your `main.py`.
- The prompts are lowercase for consistency. Cursor will remove/rename any conflicting uppercase variants if found.
- Exports save to **Dropbox** (primary) and **S3** (audit pack).

If you need a different folder layout, tell me and I’ll regenerate this pack.
