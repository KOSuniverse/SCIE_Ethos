# PY Files/phase5_governance/__init__.py
# Phase 5: Governance Lite & Data Gaps package initialization

"""
Phase 5 — GOVERNANCE LITE & DATA GAPS

This package implements the Phase 5 requirements from README_CURSOR.md:
- 5A: Query logs JSONL (user, intent, sources, confidence, tokens, $) and simple Usage page
- 5B: Aggregate "Top Missing Data" report to /04_Data/04_Metadata/missing_fields_report.json

Key Components:
- QueryLogger: Comprehensive JSONL logging of all user interactions
- UsageDashboard: Streamlit dashboard with usage analytics and visualizations  
- DataGapAnalyzer: Missing data detection and aggregated reporting system
"""

from .query_logger import QueryLogger
from .usage_dashboard import UsageDashboard
from .data_gap_analyzer import DataGapAnalyzer

__all__ = [
    "QueryLogger",
    "UsageDashboard", 
    "DataGapAnalyzer"
]

__version__ = "1.0.0"
__phase__ = "5 - GOVERNANCE LITE & DATA GAPS"
