# Phase 0 Audit Report
**SCIE/Ethos LLM ERP-lite App**  
**Branch:** `phase-refresh-01`  
**Date:** December 18, 2025

## Executive Summary
This audit identifies critical gaps between the current codebase implementation and the requirements specified in `README_CURSOR.md`. The system has foundational components but lacks several key features required for production readiness.

## 1. DROPDOWNS STILL PRESENT (CRITICAL VIOLATION)

### Files with Dropdowns:
- **`main.py:517`** - `st.selectbox("Pick a RAW file (Excel or CSV)", options=labels, index=0)`
- **`main.py:1472`** - `st.selectbox("Pick a Cleansed file (Excel or CSV)", options=cln_labels, index=0, key="cln_pick")`
- **`main.py:1564`** - `st.selectbox("Choose a cleansed file (Excel or CSV):", range(len(files)), format_func=lambda i: nice_labels[i], key="qa_select_clean")`
- **`chat_ui.py:132`** - `st.selectbox` for service level
- **`chat_ui.py:157`** - `st.radio` for model choice
- **`data_needed_panel.py:104`** - `st.selectbox("Data Type", ["Inventory", "WIP", "E&O", "Forecast", "Other"], key="gap_type")`
- **`data_needed_panel.py:107`** - `st.selectbox("Priority", self.priority_levels, key="gap_priority")`

**Status:** ❌ **BLOCKER** - Multiple dropdowns violate the "No dropdowns" non-negotiable requirement.

## 2. SOURCES DRAWER + CONFIDENCE BADGE RENDERING

### Chat Mode (✅ IMPLEMENTED):
- **`chat_ui.py:249-250`** - Sources drawer renders with `sources_drawer.render_inline_sources(sources, confidence_score)`
- **`chat_ui.py:188-192`** - Confidence badge displays with `get_confidence_badge(last_confidence, st.session_state.service_level)`

### Data Processing Mode (❌ MISSING):
- **`main.py`** - No sources drawer or confidence badge implementation found
- **`loader.py`** - No confidence scoring integration
- **`phase1_ingest/`** - No sources tracking

**Status:** ❌ **BLOCKER** - Sources drawer and confidence badge only render in Chat mode, not in Data Processing mode.

## 3. DICT/LIST MISUSE (POTENTIAL "LIST INDICES" CRASH)

### High-Risk Patterns Found:
- **`main.py:522,549,550,1475`** - `raw_files[labels.index(choice)]` - assumes `labels.index()` returns valid index
- **`loader.py:211`** - `raw_df.iloc[hdr_idx + 1].tolist()` - no bounds checking
- **`phase2_analysis/enhanced_eda_system.py:220,235,770,785`** - Multiple `.iloc[0]` calls without length validation
- **`phase3_comparison/ranking_utils.py:233,242`** - `.iloc[0]` without bounds checking
- **`phase1_ingest/smart_cleaning.py:235,260`** - `.iloc[0]` on potentially empty DataFrames

**Status:** ❌ **HIGH RISK** - Multiple locations where list indices could cause crashes.

## 4. MULTI-FILE COMPARISON MISSING

### Current Implementation:
- **`phase3_comparison/comparison_utils.py`** - Only handles WIP aging comparison
- **`orchestrator.py:700,808,1065`** - Intent detection for comparison exists but no multi-file engine
- **No auto-pairing** of Q1/Q2 files
- **No comparison workbook** generation with required tabs (Aligned, Delta, Only_A, Only_B, Aging_Shift, Schema_Mismatch_Report, Charts_Data)

### Missing Features:
- Multi-file comparison engine
- Auto-pairing logic for time periods
- Comparison workbook export to `/04_Data/05_Merged_Comparisons/`
- Delta analysis across multiple files

**Status:** ❌ **MISSING** - Multi-file comparison functionality not implemented.

## 5. MISSING REQUIRED VISUALS

### Required Charts (from README):
- **Inventory aging waterfall** - ❌ Not implemented
- **Usage-vs-stock** - ❌ Not implemented  
- **Treemap** - ❌ Not implemented
- **Forecast vs actual** - ❌ Not implemented
- **MOQ histogram** - ❌ Not implemented

### Current Charting:
- **`PY Files/charting.py`** - Basic bar charts, line charts, scatter plots
- **`phase2_analysis/enhanced_eda_system.py`** - Basic EDA charts
- **No specialized inventory charts** matching requirements

**Status:** ❌ **MISSING** - All 5 required chart types are not implemented.

## 6. KB RETRIEVAL ENFORCEMENT

### Current Implementation:
- **`phase4_knowledge/knowledgebase_retriever.py:118`** - `search_topk()` function exists
- **`orchestrator.py:948-950`** - `top_k = int(args.get("k", 5))` - defaults to k=5, not k≥4
- **No enforcement** of minimum k=4 requirement
- **No low coverage warnings** for <2 hits

### Missing Requirements:
- Force k≥4 for all KB queries
- Low coverage warnings
- Citation validation

**Status:** ❌ **PARTIAL** - KB retrieval exists but k≥4 enforcement missing.

## 7. OUTPUT TEMPLATE COMPLIANCE

### Template Definition:
- **`configs/orchestrator_rules.yaml:28-29`** - Template defined: `["Title","Executive Insight","Analysis","Recommendations","Citations","Limits / Data Needed"]`
- **`PY Files/orchestrator.py:440-444`** - Template loading exists

### Implementation Status:
- **Template structure** defined ✅
- **Enforcement mechanism** unclear ❌
- **No validation** that answers follow template ❌

**Status:** ❌ **PARTIAL** - Template defined but enforcement unclear.

## 8. LOGGING AND CONFIDENCE SCORING

### Confidence Scoring (✅ IMPLEMENTED):
- **`PY Files/confidence.py`** - R/A/V/C scoring system exists
- **`PY Files/orchestrator.py:299`** - Confidence config integration
- **`chat_ui.py:103-115`** - Enhanced confidence badge generation

### Logging System (✅ IMPLEMENTED):
- **`PY Files/logging_system.py`** - Comprehensive logging with S3 integration
- **`PY Files/monitoring_dashboard.py`** - Monitoring dashboard exists
- **Turn-by-turn logging** implemented

**Status:** ✅ **IMPLEMENTED** - Both confidence scoring and logging systems are present.

## 9. ADDITIONAL FINDINGS

### Header Row Detection:
- **`loader.py:190-211`** - Header row detection exists with configurable scan limit
- **`_SCAN_LIMIT = 150`** - Configurable via environment variable
- **No hard-coded** header row assumptions found

### Data Gap Detection:
- **`PY Files/data_needed_panel.py`** - Data gap tracking exists
- **Integration** with main workflow unclear

## 10. PRIORITY RECOMMENDATIONS

### Phase 1 Blockers (Immediate):
1. **Remove all dropdowns** from `main.py` and other files
2. **Fix dict/list misuse** patterns to prevent crashes
3. **Implement missing visuals** (5 chart types)
4. **Add sources drawer + confidence badge** to Data Processing mode

### Phase 2 Critical:
1. **Implement multi-file comparison engine**
2. **Enforce KB retrieval k≥4**
3. **Validate output template compliance**

### Phase 3 Enhancement:
1. **Complete chart implementations**
2. **Add comparison workbook export**
3. **Enhance citation validation**

## 11. RISK ASSESSMENT

- **Critical Risk:** Dropdowns violate core requirements
- **High Risk:** List index crashes in production
- **Medium Risk:** Missing multi-file comparison
- **Low Risk:** Missing specialized charts

## 12. COMPLIANCE SCORE

- **Phase 0 Requirements:** 3/8 ✅ (37.5%)
- **Production Readiness:** ❌ **NOT READY**
- **Critical Blockers:** 3
- **High Priority Issues:** 2
- **Medium Priority Issues:** 2

---

**Next Steps:** Execute Phase 1 fixes to address critical blockers before proceeding with feature development.
