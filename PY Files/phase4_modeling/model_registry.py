# PY Files/phase4_modeling/model_registry.py
# Phase 4B: Model artifacts and metadata management

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

class ModelRegistry:
    """
    Phase 4B: Save model artifacts/metadata under `/04_Data/Models/`.
    Manages forecasting model persistence, versioning, and retrieval.
    """
    
    def __init__(self, base_path: str = "04_Data/Models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "forecasting").mkdir(exist_ok=True)
        (self.base_path / "safety_stock").mkdir(exist_ok=True)
        (self.base_path / "par_levels").mkdir(exist_ok=True)
        (self.base_path / "backtests").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
    
    def save_forecast_model(
        self,
        model_results: Dict[str, Any],
        part_number: str,
        model_type: str = "exponential_smoothing"
    ) -> str:
        """
        Save forecasting model results and metadata.
        
        Returns:
            str: Model ID for future reference
        """
        
        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type}_{part_number}_{timestamp}"
        
        # Prepare metadata
        metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "part_number": part_number,
            "created_at": datetime.now().isoformat(),
            "model_class": "forecasting",
            "version": "1.0",
            "parameters": model_results.get("model_metadata", {}),
            "performance_metrics": model_results.get("accuracy_metrics", {}),
            "service_level": model_results.get("service_level", 0.95)
        }
        
        # Save forecast data
        forecast_data = {
            "model_id": model_id,
            "forecast": model_results.get("forecast", []),
            "confidence_intervals": model_results.get("confidence_intervals", []),
            "forecast_periods": len(model_results.get("forecast", [])),
            "metadata": metadata
        }
        
        # Save files
        model_dir = self.base_path / "forecasting" / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save forecast results
        with open(model_dir / "forecast_results.json", "w") as f:
            json.dump(forecast_data, f, indent=2)
        
        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save to master registry
        self._update_master_registry(metadata)
        
        return model_id
    
    def save_policy_model(
        self,
        policy_results: Dict[str, Any],
        part_number: str,
        policy_type: str = "comprehensive"
    ) -> str:
        """
        Save inventory policy model (safety stock, ROP, par levels).
        
        Returns:
            str: Model ID for future reference
        """
        
        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"policy_{policy_type}_{part_number}_{timestamp}"
        
        # Prepare metadata
        metadata = {
            "model_id": model_id,
            "model_type": policy_type,
            "part_number": part_number,
            "created_at": datetime.now().isoformat(),
            "model_class": "inventory_policy",
            "version": "1.0",
            "current_analysis": policy_results.get("current_analysis", {}),
            "recommendations": policy_results.get("recommendations", []),
            "service_level": policy_results.get("service_level", 0.95)
        }
        
        # Extract policy parameters
        policy_data = {
            "model_id": model_id,
            "safety_stock": policy_results.get("safety_stock", {}),
            "reorder_point": policy_results.get("reorder_point", {}),
            "par_level": policy_results.get("par_level", {}),
            "demand_forecast": policy_results.get("demand_forecast", {}),
            "backtest_performance": policy_results.get("backtest_performance", {}),
            "metadata": metadata
        }
        
        # Save files
        model_dir = self.base_path / "par_levels" / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save policy results
        with open(model_dir / "policy_results.json", "w") as f:
            json.dump(policy_data, f, indent=2)
        
        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual components
        self._save_component_models(policy_results, model_dir, model_id)
        
        # Save to master registry
        self._update_master_registry(metadata)
        
        return model_id
    
    def _save_component_models(
        self, 
        policy_results: Dict[str, Any], 
        model_dir: Path, 
        model_id: str
    ):
        """Save individual model components (safety stock, ROP, etc.)."""
        
        # Safety Stock Model
        if "safety_stock" in policy_results:
            ss_dir = self.base_path / "safety_stock" / f"{model_id}_ss"
            ss_dir.mkdir(exist_ok=True)
            
            with open(ss_dir / "safety_stock_model.json", "w") as f:
                json.dump({
                    "model_id": f"{model_id}_ss",
                    "parent_model": model_id,
                    "component": "safety_stock",
                    "results": policy_results["safety_stock"],
                    "created_at": datetime.now().isoformat()
                }, f, indent=2)
        
        # Backtest Results
        if "backtest_performance" in policy_results:
            bt_dir = self.base_path / "backtests" / f"{model_id}_backtest"
            bt_dir.mkdir(exist_ok=True)
            
            backtest_data = policy_results["backtest_performance"]
            
            # Save as JSON
            with open(bt_dir / "backtest_results.json", "w") as f:
                json.dump({
                    "model_id": f"{model_id}_backtest",
                    "parent_model": model_id,
                    "component": "backtest",
                    "results": backtest_data,
                    "created_at": datetime.now().isoformat()
                }, f, indent=2)
            
            # Save as CSV for easy analysis
            if "backtest_results" in backtest_data:
                bt_results = backtest_data["backtest_results"]
                if bt_results.get("actuals") and bt_results.get("forecasts"):
                    df = pd.DataFrame({
                        "period": range(1, len(bt_results["actuals"]) + 1),
                        "actual": bt_results["actuals"],
                        "forecast": bt_results["forecasts"],
                        "error": bt_results["errors"]
                    })
                    df.to_csv(bt_dir / "backtest_data.csv", index=False)
    
    def _update_master_registry(self, metadata: Dict[str, Any]):
        """Update the master model registry file."""
        
        registry_file = self.base_path / "metadata" / "model_registry.json"
        
        # Load existing registry
        if registry_file.exists():
            with open(registry_file, "r") as f:
                registry = json.load(f)
        else:
            registry = {
                "models": [],
                "last_updated": None,
                "total_models": 0
            }
        
        # Add new model
        registry["models"].append(metadata)
        registry["last_updated"] = datetime.now().isoformat()
        registry["total_models"] = len(registry["models"])
        
        # Save updated registry
        with open(registry_file, "w") as f:
            json.dump(registry, f, indent=2)
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a model by ID."""
        
        # Search in different model types
        search_paths = [
            self.base_path / "forecasting" / model_id,
            self.base_path / "par_levels" / model_id,
            self.base_path / "safety_stock" / model_id,
            self.base_path / "backtests" / model_id
        ]
        
        for path in search_paths:
            if path.exists():
                # Try to load main results file
                result_files = [
                    "forecast_results.json",
                    "policy_results.json", 
                    "safety_stock_model.json",
                    "backtest_results.json"
                ]
                
                for result_file in result_files:
                    file_path = path / result_file
                    if file_path.exists():
                        with open(file_path, "r") as f:
                            return json.load(f)
        
        return None
    
    def list_models(
        self, 
        model_class: Optional[str] = None,
        part_number: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List models with optional filtering."""
        
        registry_file = self.base_path / "metadata" / "model_registry.json"
        
        if not registry_file.exists():
            return []
        
        with open(registry_file, "r") as f:
            registry = json.load(f)
        
        models = registry.get("models", [])
        
        # Apply filters
        if model_class:
            models = [m for m in models if m.get("model_class") == model_class]
        
        if part_number:
            models = [m for m in models if m.get("part_number") == part_number]
        
        return models
    
    def get_latest_model(
        self, 
        part_number: str, 
        model_class: str = "inventory_policy"
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent model for a part."""
        
        models = self.list_models(model_class=model_class, part_number=part_number)
        
        if not models:
            return None
        
        # Sort by creation date and get latest
        latest = max(models, key=lambda x: x.get("created_at", ""))
        return self.get_model(latest["model_id"])
    
    def create_model_summary_report(self) -> Dict[str, Any]:
        """Create a summary report of all models in the registry."""
        
        registry_file = self.base_path / "metadata" / "model_registry.json"
        
        if not registry_file.exists():
            return {"error": "No models found in registry"}
        
        with open(registry_file, "r") as f:
            registry = json.load(f)
        
        models = registry.get("models", [])
        
        # Summary statistics
        summary = {
            "total_models": len(models),
            "model_classes": {},
            "model_types": {},
            "parts_covered": set(),
            "latest_model": None,
            "oldest_model": None
        }
        
        for model in models:
            # Count by class
            model_class = model.get("model_class", "unknown")
            summary["model_classes"][model_class] = summary["model_classes"].get(model_class, 0) + 1
            
            # Count by type
            model_type = model.get("model_type", "unknown")
            summary["model_types"][model_type] = summary["model_types"].get(model_type, 0) + 1
            
            # Track parts
            part = model.get("part_number")
            if part:
                summary["parts_covered"].add(part)
        
        # Convert set to list for JSON serialization
        summary["parts_covered"] = list(summary["parts_covered"])
        summary["unique_parts"] = len(summary["parts_covered"])
        
        # Find latest and oldest
        if models:
            summary["latest_model"] = max(models, key=lambda x: x.get("created_at", ""))["model_id"]
            summary["oldest_model"] = min(models, key=lambda x: x.get("created_at", ""))["model_id"]
        
        return summary
    
    def export_model_catalog(self, output_path: str = None) -> str:
        """Export a comprehensive model catalog as Excel file."""
        
        if not output_path:
            output_path = f"model_catalog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        models = self.list_models()
        
        if not models:
            return "No models found to export"
        
        # Create DataFrame
        catalog_data = []
        for model in models:
            catalog_data.append({
                "Model ID": model.get("model_id"),
                "Part Number": model.get("part_number"),
                "Model Class": model.get("model_class"),
                "Model Type": model.get("model_type"),
                "Created At": model.get("created_at"),
                "Service Level": model.get("service_level"),
                "Version": model.get("version")
            })
        
        df = pd.DataFrame(catalog_data)
        
        # Save to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Model_Catalog', index=False)
            
            # Add summary sheet
            summary = self.create_model_summary_report()
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return output_path
