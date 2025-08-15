# Cursor Build – SCIE Ethos (End-State)

## Goals
Build a cloud-only Streamlit app backed by OpenAI Assistants API (File Search + Code Interpreter), with Dropbox as primary storage and S3 for logs/exports. The system detects issues in inventory/WIP/E&O, finds root causes, and produces implementable, citation-backed solutions and forecasts (incl. Par/ROP now and to future horizons).

## Inputs in this repo
- /prompts/instructions_master.yaml  (core brain + guardrails)
- /prompts/glossary_aliases.yaml     (EN/IT/DE header & sheet aliases)
- /prompts/modeling_playbooks.yaml   (predictive modeling playbooks)
- /prompts/ingest_rules.yaml         (tagging tiers, sheet-splitting, artifacts)
- /schemas/master_catalog.schema.json
- /schemas/s3_turn_log.schema.json
- /templates/summary_card.template.md
- /configs/export_policy.yaml
- /configs/retention_policy.yaml
- /configs/orchestrator_rules.yaml
- /qa/acceptance_suite.yaml
- UI_SPEC_ADDENDUM.md


## Pre-step: Audit & Refactor Existing Repo
- Scan current repository (branch: `cursor-endstate`) and produce a short plan with a file-by-file diff against the structure above.
- **Refactor/rename** any existing files to align with this layout (e.g., `app/ui_main.py` → `main.py`; prompts `INSTRUCTIONS_MASTER.yaml` → `instructions_master.yaml`, `GLOSSARY_ALIASES.yaml` → `glossary_aliases.yaml`).
- Preserve useful code (connectors, sync manager, storage providers). Update them to adhere to `ingest_rules.yaml` and emit artifacts (`master_catalog.jsonl`, `eda_profile.json`, `summary_card.md`).
- Remove/merge duplicates and dead code. Commit refactors with clear messages.

## Build Tasks
1. **Create Assistant**
   - Enable **File Search** and **Code Interpreter**.
   - Load `instructions_master.yaml`, `glossary_aliases.yaml`, `modeling_playbooks.yaml`.
2. **Dropbox → Assistant Sync**
   - Implement a sync tool: `sync --root <dropbox_path> --assistant <id> --dry-run`.
   - Apply **ingest_rules.yaml** to create per-sheet “virtual files”, compute Tier-0…Tier-4 tags, and generate artifacts:
     - `master_catalog.jsonl` (append-only)
     - `eda_profile.json` (per virtual file)
     - `summary_card.md` from template
   - Upload artifacts + source sheets to Assistant File Store.
   - Emit an `ingest_run_<ts>.jsonl` report to S3.
3. **Streamlit App (cloud)**
   - `main.py`: ChatGPT-like UI with streaming, Sources drawer (clickable citations), Confidence badge, “Data Needed” panel, and export buttons.
   - Service-Level control in UI (90/95/97.5/99) mapped to z-score; persist selection per session.
   - Exports: Dropbox primary (per `export_policy.yaml`), S3 audit pack.
4. **Orchestrator**
   - Implement routing, retrieval, coverage check, dual-pass verify, escalation, scoring per `orchestrator_rules.yaml` + `instructions_master.yaml`.
   - Use `master_catalog.jsonl` and `summary_card.md` to prioritize tight top-k retrieval.
5. **Logging & Retention**
   - Write one record per turn to S3 matching `s3_turn_log.schema.json`.
   - Apply `retention_policy.yaml` to logs & export packs.
6. **QA**
   - Run `/qa/acceptance_suite.yaml`. All tests must return citations; ≥80% with confidence ≥0.70; any abstentions must list precise missing data.


## Folder Output Contract (MUST HONOR)
- Read raw data from `${base_data_dir}/${folders.raw}`.
- Write cleansed to `${base_data_dir}/${folders.cleansed}`; never overwrite raw.
- Write metadata artifacts to `${base_data_dir}/${folders.metadata}`.
- Write summaries to `${base_data_dir}/${folders.summaries}`; EDA charts to `${base_data_dir}/${folders.eda_charts}`.
- Write roll-ups to `${base_data_dir}/${folders.merged}`.
- Exports to `${folders.exports}`; logs to `${folders.logs}`.
- Paths are defined in `/configs/path_contract.yaml`. Do not invent new folders.

## Guardrails (enforced)
- No Source? No Answer. If citations fail or sim score < threshold, return “Insufficient evidence” + Data Needed.
- No dropdowns. Mode selection is internal via intent routing.
- Strict output template: Title → Executive Insight → Analysis → Recommendations → Citations → Limits / Data Needed.
- Auto planning hooks: all RCA/movement answers include policy/risk/action items.
- Data gap logging (S3) and token budgeting with pagination.
