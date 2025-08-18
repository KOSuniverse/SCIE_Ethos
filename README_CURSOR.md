You are working on our SCIE/Ethos LLM ERP-lite app.

MISSION
1) **Audit** all code changes you’ve made previously, detect regressions and missed items against the spec below.
2) **Fix in phases** (small PRs with tests), producing a report per phase of what was missing and what you fixed.

NON-NEGOTIABLES
- **No dropdowns**. All analysis is natural-language with **auto intent routing**.
- **KB citations** on every material claim.
- **Specificity**: show SKUs/vendors/locations and $$ deltas; no vague “costs are high”.
- **Visuals**: auto-generate and persist required charts.
- **Comparison mode**: must handle 2+ files (Q1 vs Q2 etc.), save a comparison workbook with lineage & deltas.
- **EDA robustness**: fix “list indices” crash; header rows at 13/15; alias map; never drop rows silently.
- **Confidence**: show High/Med/Low + score; escalate to stronger model if Low.

CODEBASE FOCUS (expected files; adjust to our repo names)
- `ui_main.py` (Streamlit shell) — remove dropdowns; add sources drawer, confidence badge.
- `orchestrator.py` — intent routing; retrieval; verification; escalation.
- `prompts/instructions_master.yaml` — intents & sub-skills (root_cause, forecasting→par_policy/safety_stock/demand_projection, movement_analysis, optimization, scenario, exec_summary, gap_check, eo_analysis, wip_root_cause).
- `prompts/glossary_aliases.yaml` — domain glossary + column/sheet alias map.
- `data/compare.py` or similar — multi-file compare (delta/aging-shift).
- `eda/normalization.py` — header sniff (13/15), aliasing, sheet_type normalization.
- `charts/` — auto charts (inventory aging waterfall, usage-vs-stock, treemap, forecast vs actual, MOQ histogram).
- `confidence.py` — R/A/V/C scoring; thresholds.
- `logging_utils.py` — query log JSONL (user, intent, sources, tokens, $).
- `dropbox_sync.py` (or connector) — ingest & lineage tags.
- `tests/` — add fixtures and CI.

GLOBAL REQUIREMENTS
- **Auto intent** only. Kill `st.selectbox`/`st.radio` paths.
- **Comparison**: auto-pair files (Q1/Q2; date tokens) and allow 2+ files; output workbook saved to `/04_Data/05_Merged_Comparisons/{timestamp}_comparison.xlsx` with tabs: Aligned, Delta, Only_A, Only_B, Aging_Shift (if WIP), Schema_Mismatch_Report, Charts_Data.
- **Charts** saved to `/04_Data/02_EDA_Charts/` and embedded in UI.
- **Lineage columns**: `source_file`, `source_sheet`, `header_row`, `sheet_type`.
- **Data Gap Detection** in every answer: list missing fields required for higher confidence.
- **KB Retrieval**: force k≥4; show doc titles and sections in Sources drawer; warn on low coverage.
- **Predictive modeling**: par/ROP/safety stock computation; backtests; write model metadata to `/04_Data/Models/`.
- **Outputs follow template**: Title → Executive Insight → Detailed Analysis (tables/charts) → Recommendations → Citations → Limits/Missing Data.

PHASE PLAN (execute in order; each is a separate PR)

PHASE 0 — AUDIT (no code changes)
Deliver `/audit/phase0_report.md`:
- Where dropdowns still exist (file:line).
- All places referencing dict keys on lists (root cause of “list indices” crash).
- Files where header row sniff is hard-coded or missing.
- Missing visuals and where chart calls are stubbed.
- Where KB retrieval is not enforced (k<4 or not called).
- Where answers don’t follow the output template.
- Where comparison code handles only a single file.
- Any missing logging, confidence scoring, or data gap flags.

PHASE 1 — BLOCKERS
1A. Remove dropdowns entirely; enforce auto-intent; add hidden debug caption `Intent: <intent>`.
1B. Fix EDA robustness: guard list/dict cases; header sniff 13/15; alias map application; no silent row drops (fill NA + log).
1C. Visuals: implement and persist the 5 core charts; render in UI.
Acceptance:
- 10 natural-language queries run with no dropdowns.
- Large sheet with header at row 13 loads without crash.
- At least 5 charts saved to `/02_EDA_Charts/` and visible.

PHASE 2 — COMPARISON ENGINE
2A. Implement multi-file compare (2+ files), auto-pairing (Q1/Q2 tokens), key detection (`plant`, `part_number|kit_number`, `job_number`).
2B. Produce Delta, Only_A, Only_B, Aging_Shift, Schema_Mismatch_Report; persist a comparison workbook to `/05_Merged_Comparisons/`.
2C. Add comparison-aware visuals (delta waterfall, aging shift, movers scatter).
Acceptance:
- Asking “Compare Q1 vs Q2 aging and show top movers” produces tables + charts + saved workbook.
- Debug caption shows chosen keys and auto-paired files.

PHASE 3 — ANSWER QUALITY, KB, CONFIDENCE
3A. Enforce output template; require ≥2 identifiers per claim; driver table with $$ impacts.
3B. Force KB retrieval (k≥4), show citations; low coverage warning if <2 hits.
3C. Confidence scoring (R/A/V/C) + escalation to stronger model if Low; show badge.
Acceptance:
- Answers display template sections; include specific SKUs/vendors/locations with $$ deltas and citations.
- Confidence badge present with numeric score; low-confidence queries escalate.

PHASE 4 — MODELING + REGISTRY
4A. Forecasting sub-skills (par_policy, safety_stock, demand_projection) with backtests and policy math (SS, ROP, Par).
4B. Save model artifacts/metadata under `/04_Data/Models/`.
4C. Add “Buy list / policy change” XLSX export.
Acceptance:
- “Par level for Family X in Germany next quarter” yields tables, chart, metadata, and an export.

PHASE 5 — GOVERNANCE LITE & DATA GAPS
5A. Query logs JSONL (user, intent, sources, confidence, tokens, $); simple Usage page.
5B. Aggregate “Top Missing Data” report to `/04_Data/04_Metadata/missing_fields_report.json`.
Acceptance:
- Usage counters render; missing-data report generated after several queries.

DELIVERABLES PER PR
- Code changes with **unit tests** (fixtures for two tiny Q1/Q2 files).
- **Screenshots/GIFs** of UI paths exercised.
- Short **CHANGELOG.md** entry and updated **README** section.
- A brief **smoke script** in `/scripts/smoke_test.sh` with 5–7 commands to reproduce.

AFTER EACH PHASE
- Write `/audit/phaseX_findings.md`: list items you fixed and anything still missing with a plan.
- Run `scripts/smoke_test.sh` and attach the output.
