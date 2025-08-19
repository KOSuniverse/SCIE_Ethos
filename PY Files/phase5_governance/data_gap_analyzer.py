# PY Files/phase5_governance/data_gap_analyzer.py
# Phase 5B: Aggregate "Top Missing Data" report

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

class DataGapAnalyzer:
    """
    Phase 5B: Aggregate "Top Missing Data" report to /04_Data/04_Metadata/missing_fields_report.json.
    Analyzes queries and data processing to identify commonly missing fields and data gaps.
    """
    
    def __init__(self, report_path: str = "04_Data/04_Metadata/missing_fields_report.json"):
        self.report_path = Path(report_path)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Common required fields by data type
        self.required_fields = {
            "inventory": {
                "critical": ["part_number", "quantity", "unit_cost", "location"],
                "important": ["description", "supplier", "lead_time", "reorder_point", "safety_stock"],
                "useful": ["category", "abc_class", "last_movement_date", "usage_rate"]
            },
            "wip": {
                "critical": ["job_number", "part_number", "wip_value", "job_status"],
                "important": ["start_date", "due_date", "customer", "priority"],
                "useful": ["operation", "work_center", "setup_time", "run_time"]
            },
            "financial": {
                "critical": ["account_number", "amount", "transaction_date", "currency"],
                "important": ["description", "cost_center", "department", "gl_account"],
                "useful": ["reference_number", "approval_status", "budget_code"]
            },
            "demand": {
                "critical": ["part_number", "demand_quantity", "period", "location"],
                "important": ["forecast_method", "confidence_level", "seasonality"],
                "useful": ["customer_segment", "product_family", "market_region"]
            }
        }
        
        # Initialize gap tracking
        self.gap_history = []
    
    def analyze_data_gaps(
        self,
        data_sources: List[Dict[str, Any]],
        query_intent: str,
        user_query: str,
        analysis_results: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze data gaps in the provided data sources for a specific query.
        
        Args:
            data_sources: List of data source dictionaries with metadata
            query_intent: Intent of the user query (inventory, wip, forecast, etc.)
            user_query: Original user query text
            analysis_results: Results from data processing (optional)
        
        Returns:
            Dict with identified gaps and recommendations
        """
        
        timestamp = datetime.now().isoformat()
        
        # Determine data type from intent
        data_type = self._map_intent_to_data_type(query_intent)
        
        # Analyze each data source
        source_gaps = []
        overall_gaps = {
            "missing_critical": set(),
            "missing_important": set(), 
            "missing_useful": set(),
            "data_quality_issues": []
        }
        
        for source in data_sources:
            gap_analysis = self._analyze_single_source(source, data_type)
            source_gaps.append(gap_analysis)
            
            # Aggregate gaps
            overall_gaps["missing_critical"].update(gap_analysis["missing_critical"])
            overall_gaps["missing_important"].update(gap_analysis["missing_important"])
            overall_gaps["missing_useful"].update(gap_analysis["missing_useful"])
            overall_gaps["data_quality_issues"].extend(gap_analysis["data_quality_issues"])
        
        # Convert sets to lists for JSON serialization
        overall_gaps["missing_critical"] = list(overall_gaps["missing_critical"])
        overall_gaps["missing_important"] = list(overall_gaps["missing_important"])
        overall_gaps["missing_useful"] = list(overall_gaps["missing_useful"])
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(overall_gaps, query_intent)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_gaps, data_type, query_intent)
        
        # Create gap analysis result
        gap_result = {
            "timestamp": timestamp,
            "query": {
                "text": user_query,
                "intent": query_intent,
                "data_type": data_type
            },
            "sources_analyzed": len(data_sources),
            "gaps_identified": overall_gaps,
            "source_details": source_gaps,
            "impact_score": impact_score,
            "recommendations": recommendations,
            "confidence_impact": self._assess_confidence_impact(overall_gaps)
        }
        
        # Store for aggregation
        self.gap_history.append(gap_result)
        
        return gap_result
    
    def _map_intent_to_data_type(self, intent: str) -> str:
        """Map query intent to data type for gap analysis."""
        
        intent_mapping = {
            "forecast": "demand",
            "eda": "inventory",
            "compare": "inventory", 
            "root_cause": "inventory",
            "movement_analysis": "inventory",
            "eo_analysis": "inventory",
            "optimize": "inventory",
            "scenario": "inventory",
            "kb_lookup": "general"
        }
        
        return intent_mapping.get(intent.lower(), "inventory")
    
    def _analyze_single_source(self, source: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Analyze gaps in a single data source."""
        
        source_path = source.get("path", "unknown")
        source_columns = source.get("columns", [])
        source_rows = source.get("rows", 0)
        
        if data_type not in self.required_fields:
            data_type = "inventory"  # Default fallback
        
        required = self.required_fields[data_type]
        
        # Check for missing fields
        missing_critical = []
        missing_important = []
        missing_useful = []
        
        for field in required["critical"]:
            if not self._field_exists(field, source_columns):
                missing_critical.append(field)
        
        for field in required["important"]:
            if not self._field_exists(field, source_columns):
                missing_important.append(field)
        
        for field in required["useful"]:
            if not self._field_exists(field, source_columns):
                missing_useful.append(field)
        
        # Check data quality issues
        quality_issues = []
        
        if source_rows == 0:
            quality_issues.append("No data rows found")
        elif source_rows < 10:
            quality_issues.append(f"Very small dataset ({source_rows} rows)")
        
        if len(source_columns) < 3:
            quality_issues.append("Very few columns available")
        
        return {
            "source_path": source_path,
            "columns_available": len(source_columns),
            "rows_available": source_rows,
            "missing_critical": missing_critical,
            "missing_important": missing_important,
            "missing_useful": missing_useful,
            "data_quality_issues": quality_issues,
            "completeness_score": self._calculate_completeness_score(
                missing_critical, missing_important, missing_useful, required
            )
        }
    
    def _field_exists(self, field: str, columns: List[str]) -> bool:
        """Check if a field exists in columns (with fuzzy matching)."""
        
        if not columns:
            return False
        
        # Direct match
        if field in columns:
            return True
        
        # Case-insensitive match
        field_lower = field.lower()
        for col in columns:
            if col.lower() == field_lower:
                return True
        
        # Partial match for common variations
        field_variations = {
            "part_number": ["part", "item", "sku", "product", "material"],
            "quantity": ["qty", "amount", "count", "units"],
            "unit_cost": ["cost", "price", "value", "unit_price"],
            "location": ["loc", "warehouse", "site", "plant"],
            "job_number": ["job", "order", "work_order", "wo"],
            "wip_value": ["wip", "work_in_progress", "value"],
            "account_number": ["account", "gl_account", "account_code"]
        }
        
        variations = field_variations.get(field, [field])
        
        for variation in variations:
            for col in columns:
                if variation.lower() in col.lower():
                    return True
        
        return False
    
    def _calculate_completeness_score(
        self, 
        missing_critical: List[str], 
        missing_important: List[str], 
        missing_useful: List[str],
        required: Dict[str, List[str]]
    ) -> float:
        """Calculate completeness score (0-100) for a data source."""
        
        total_critical = len(required["critical"])
        total_important = len(required["important"])
        total_useful = len(required["useful"])
        
        critical_score = (total_critical - len(missing_critical)) / total_critical if total_critical > 0 else 1
        important_score = (total_important - len(missing_important)) / total_important if total_important > 0 else 1
        useful_score = (total_useful - len(missing_useful)) / total_useful if total_useful > 0 else 1
        
        # Weighted score (critical=50%, important=30%, useful=20%)
        weighted_score = (critical_score * 0.5) + (important_score * 0.3) + (useful_score * 0.2)
        
        return round(weighted_score * 100, 1)
    
    def _calculate_impact_score(self, gaps: Dict[str, Any], intent: str) -> int:
        """Calculate impact score (1-10) based on missing data severity."""
        
        critical_count = len(gaps["missing_critical"])
        important_count = len(gaps["missing_important"])
        quality_issues = len(gaps["data_quality_issues"])
        
        # Base impact calculation
        impact = 0
        
        # Critical fields have highest impact
        impact += critical_count * 3
        
        # Important fields have moderate impact  
        impact += important_count * 2
        
        # Quality issues add to impact
        impact += quality_issues * 1
        
        # Intent-specific adjustments
        high_impact_intents = ["forecast", "optimize", "scenario"]
        if intent in high_impact_intents:
            impact = int(impact * 1.5)
        
        # Cap at 10
        return min(impact, 10)
    
    def _generate_recommendations(
        self, 
        gaps: Dict[str, Any], 
        data_type: str, 
        intent: str
    ) -> List[str]:
        """Generate actionable recommendations for addressing data gaps."""
        
        recommendations = []
        
        # Critical field recommendations
        if gaps["missing_critical"]:
            recommendations.append(
                f"🔴 CRITICAL: Add missing essential fields: {', '.join(gaps['missing_critical'][:3])}"
            )
            
            # Specific recommendations by field
            for field in gaps["missing_critical"][:3]:
                if field == "part_number":
                    recommendations.append("Add unique part/SKU identifiers to enable inventory tracking")
                elif field == "quantity":
                    recommendations.append("Include quantity/amount columns for accurate calculations")
                elif field == "unit_cost":
                    recommendations.append("Add cost/price data for financial analysis")
                elif field == "job_number":
                    recommendations.append("Include job/order numbers for WIP tracking")
        
        # Important field recommendations
        if gaps["missing_important"]:
            recommendations.append(
                f"🟡 IMPORTANT: Consider adding: {', '.join(gaps['missing_important'][:3])}"
            )
        
        # Intent-specific recommendations
        if intent == "forecast":
            recommendations.append("For forecasting: Historical demand data and seasonality indicators improve accuracy")
        elif intent == "optimize":
            recommendations.append("For optimization: Lead times, safety stock levels, and supplier data are valuable")
        elif intent == "root_cause":
            recommendations.append("For root cause analysis: Timestamps, status changes, and operational metrics help")
        
        # Data quality recommendations
        quality_issues = gaps.get("data_quality_issues", [])
        if quality_issues:
            recommendations.append(f"📊 DATA QUALITY: {quality_issues[0]}")
        
        return recommendations
    
    def _assess_confidence_impact(self, gaps: Dict[str, Any]) -> str:
        """Assess how data gaps impact confidence scores."""
        
        critical_count = len(gaps["missing_critical"])
        important_count = len(gaps["missing_important"])
        
        if critical_count >= 2:
            return "High impact - Missing critical fields significantly reduce analysis confidence"
        elif critical_count == 1 or important_count >= 3:
            return "Medium impact - Some key data missing, confidence moderately affected"
        elif important_count > 0:
            return "Low impact - Minor gaps, confidence slightly reduced"
        else:
            return "Minimal impact - All essential data available"
    
    def aggregate_missing_data_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate aggregated missing data report from recent gap analyses.
        
        Args:
            days: Number of days to include in aggregation
        
        Returns:
            Comprehensive missing data report
        """
        
        # Filter recent gap analyses
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_gaps = []
        
        for gap in self.gap_history:
            try:
                gap_time = datetime.fromisoformat(gap["timestamp"]).timestamp()
                if gap_time >= cutoff_date:
                    recent_gaps.append(gap)
            except (ValueError, KeyError):
                continue
        
        if not recent_gaps:
            return {
                "error": f"No gap analyses found in the last {days} days",
                "total_analyses": 0
            }
        
        # Aggregate missing fields
        all_missing_critical = []
        all_missing_important = []
        all_missing_useful = []
        intent_gaps = defaultdict(list)
        data_type_gaps = defaultdict(list)
        impact_scores = []
        
        for gap in recent_gaps:
            gaps = gap["gaps_identified"]
            intent = gap["query"]["intent"]
            data_type = gap["query"]["data_type"]
            
            all_missing_critical.extend(gaps["missing_critical"])
            all_missing_important.extend(gaps["missing_important"])
            all_missing_useful.extend(gaps["missing_useful"])
            
            intent_gaps[intent].extend(gaps["missing_critical"] + gaps["missing_important"])
            data_type_gaps[data_type].extend(gaps["missing_critical"] + gaps["missing_important"])
            
            impact_scores.append(gap["impact_score"])
        
        # Count occurrences
        critical_counter = Counter(all_missing_critical)
        important_counter = Counter(all_missing_important)
        useful_counter = Counter(all_missing_useful)
        
        # Create top missing data lists
        top_missing_critical = critical_counter.most_common(10)
        top_missing_important = important_counter.most_common(10)
        top_missing_useful = useful_counter.most_common(10)
        
        # Calculate statistics
        avg_impact_score = sum(impact_scores) / len(impact_scores) if impact_scores else 0
        total_queries_affected = len(recent_gaps)
        
        # Generate report
        report = {
            "report_generated": datetime.now().isoformat(),
            "analysis_period_days": days,
            "total_analyses": total_queries_affected,
            "summary": {
                "avg_impact_score": round(avg_impact_score, 1),
                "high_impact_queries": len([s for s in impact_scores if s >= 7]),
                "medium_impact_queries": len([s for s in impact_scores if 4 <= s < 7]),
                "low_impact_queries": len([s for s in impact_scores if s < 4])
            },
            "top_missing_fields": {
                "critical": [{"field": field, "frequency": count, "impact": "High"} for field, count in top_missing_critical],
                "important": [{"field": field, "frequency": count, "impact": "Medium"} for field, count in top_missing_important],
                "useful": [{"field": field, "frequency": count, "impact": "Low"} for field, count in top_missing_useful]
            },
            "gaps_by_intent": {
                intent: dict(Counter(gaps).most_common(5))
                for intent, gaps in intent_gaps.items()
            },
            "gaps_by_data_type": {
                data_type: dict(Counter(gaps).most_common(5))
                for data_type, gaps in data_type_gaps.items()
            },
            "recommendations": self._generate_aggregate_recommendations(
                top_missing_critical, top_missing_important, intent_gaps
            )
        }
        
        return report
    
    def _generate_aggregate_recommendations(
        self, 
        top_critical: List[tuple], 
        top_important: List[tuple],
        intent_gaps: Dict[str, List[str]]
    ) -> List[str]:
        """Generate recommendations based on aggregated gap analysis."""
        
        recommendations = []
        
        # Top critical field recommendations
        if top_critical:
            top_field, frequency = top_critical[0]
            recommendations.append(
                f"🔴 PRIORITY 1: '{top_field}' missing in {frequency} analyses - Add this field to improve system accuracy"
            )
        
        if len(top_critical) > 1:
            second_field, frequency = top_critical[1]
            recommendations.append(
                f"🟡 PRIORITY 2: '{second_field}' missing in {frequency} analyses - Important for comprehensive analysis"
            )
        
        # Intent-specific recommendations
        most_affected_intent = max(intent_gaps.keys(), key=lambda k: len(intent_gaps[k])) if intent_gaps else None
        if most_affected_intent:
            recommendations.append(
                f"📊 FOCUS AREA: '{most_affected_intent}' queries most affected by missing data - Review data requirements"
            )
        
        # General recommendations
        recommendations.extend([
            "🔧 PROCESS: Implement data validation rules to catch missing fields early",
            "📋 STANDARDS: Create data quality checklists for common analysis types",
            "🎯 TRAINING: Educate users on required fields for different query types"
        ])
        
        return recommendations
    
    def save_missing_data_report(self, report: Dict[str, Any] = None) -> str:
        """Save the missing data report to the specified path."""
        
        if report is None:
            report = self.aggregate_missing_data_report()
        
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        return str(self.report_path)
    
    def load_existing_gaps(self) -> bool:
        """Load existing gap history from previous sessions."""
        
        # This would typically load from a persistent store
        # For now, return False indicating no existing data
        return False
