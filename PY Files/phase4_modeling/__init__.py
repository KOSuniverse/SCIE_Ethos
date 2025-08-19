# PY Files/phase4_modeling/__init__.py
# Phase 4: Modeling + Registry package initialization

"""
Phase 4 — MODELING + REGISTRY

This package implements the Phase 4 requirements from README_CURSOR.md:
- 4A: Forecasting sub-skills (par_policy, safety_stock, demand_projection) with backtests
- 4B: Model artifacts/metadata management under /04_Data/Models/  
- 4C: Buy list / policy change XLSX export functionality

Key Components:
- ForecastingEngine: Demand forecasting with multiple methods and backtesting
- ModelRegistry: Persistent model storage and retrieval system
- PolicyExportManager: Excel export for buy lists and policy recommendations
"""

from .forecasting_engine import ForecastingEngine
from .model_registry import ModelRegistry  
from .policy_export import PolicyExportManager

__all__ = [
    "ForecastingEngine",
    "ModelRegistry", 
    "PolicyExportManager"
]

__version__ = "1.0.0"
__phase__ = "4 - MODELING + REGISTRY"
