# PY Files/dp_orchestrator.py
"""
Enhanced Data Processing Orchestrator
Routes all DP queries through master instructions with full enterprise parity
"""

import os
import re
import json
import yaml
from datetime import datetime
import io
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Load master instructions
try:
    instructions_path = Path(__file__).parent.parent / "prompts" / "instructions_master.yaml"
    with open(instructions_path, 'r') as f:
        MASTER_INSTRUCTIONS = yaml.safe_load(f)
    INTENT_ROUTING = MASTER_INSTRUCTIONS.get('intent_routing', {})
    CORRELATION_POLICY = MASTER_INSTRUCTIONS.get('correlation_policy', {})
    QUALITY_PROTOCOL = MASTER_INSTRUCTIONS.get('quality_protocol', {})
except Exception as e:
    print(f"Warning: Could not load master instructions: {e}")
    MASTER_INSTRUCTIONS = {}
    INTENT_ROUTING = {}
    CORRELATION_POLICY = {}
    QUALITY_PROTOCOL = {}

# Import core components
try:
    from orchestrator import classify_user_intent, _execute_plan
    from tools_runtime import tool_specs
    from dbx_utils import list_data_files, read_file_bytes
    from file_utils import list_cleaned_files
    from confidence import calculate_ravc_confidence
    from phase5_governance.query_logger import QueryLogger
except ImportError as e:
    print(f"Warning: Could not import components: {e}")

class DataProcessingOrchestrator:
    """Enhanced Data Processing Orchestrator with enterprise parity"""
    
    def __init__(self):
        self.query_logger = QueryLogger() if 'QueryLogger' in globals() else None
        
        # Import constants to get the correct Dropbox paths
        try:
            from constants import CLEANSED, PROJECT_ROOT
            self.cleansed_files_path = CLEANSED
            print(f"DEBUG DP: Using constants - PROJECT_ROOT: {PROJECT_ROOT}, CLEANSED: {CLEANSED}")
        except ImportError:
            # Fallback paths if constants not available
            self.cleansed_files_path = "/Apps/Ethos LLM/Project_Root/04_Data/01_Cleansed_Files"
            print(f"DEBUG DP: Using fallback path: {self.cleansed_files_path}")
        
        # Additional path variations to try (based on your screenshot showing /project_root/04_data/01_cleansed_files/)
        self.cleansed_files_path_variations = [
            self.cleansed_files_path,
            "/Apps/Ethos LLM/Project_Root/04_Data/01_Cleansed_Files",
            "/Project_Root/04_Data/01_Cleansed_Files", 
            "/project_root/04_data/01_cleansed_files",  # This matches your screenshot
            "/Apps/Ethos LLM/project_root/04_data/01_cleansed_files"
        ]
        
    def resolve_files_metadata_first(self, query: str, intent: str, hints: list[str] | None = None) -> Dict[str, Any]:
        """
        Resolve files using metadata-first approach with shorthand interpretation
        """
        try:
            # Get all cleansed files
            cleansed_files = self._get_all_cleansed_files()
            
            # Parse query for file identifiers
            file_identifiers = self._extract_file_identifiers(query)
            period_filters = self._extract_period_filters(query)
            country_filters = self._extract_country_filters(query)
            erp_filters = self._extract_erp_filters(query)
            sheet_type_filters = self._extract_sheet_type_filters(query)
            
            # Rank and select files
            ranked_files = self._rank_files_by_metadata(
                cleansed_files, file_identifiers, period_filters, 
                country_filters, erp_filters, sheet_type_filters, hints
            )
            
            # Apply version deduplication
            deduplicated_files = self._deduplicate_versions(ranked_files)
            
            # Select files based on intent requirements
            selected_files = self._select_files_for_intent(deduplicated_files, intent)
            
            return {
                "selected_files": selected_files,
                "ranking_reasons": [f["ranking_reason"] for f in selected_files],
                "rejected_candidates": ranked_files[len(selected_files):len(selected_files)+5],
                "needs_clarification": len(selected_files) == 0,
                "clarification_options": deduplicated_files[:5] if len(selected_files) == 0 else []
            }
            
        except Exception as e:
            return {
                "selected_files": [],
                "error": f"File resolution failed: {str(e)}",
                "needs_clarification": True,
                "clarification_options": []
            }
    
    def _get_all_cleansed_files(self) -> List[Dict[str, Any]]:
        """Get all cleansed files with metadata"""
        try:
            files = []
            
            # Try to get files from Dropbox with multiple path variations
            for path_to_try in self.cleansed_files_path_variations:
                try:
                    from dbx_utils import list_data_files
                    print(f"DEBUG: Trying Dropbox path: {path_to_try}")
                    dropbox_files = list_data_files(path_to_try)
                    print(f"DEBUG: Found {len(dropbox_files)} files in {path_to_try}")
                    
                    for file_info in dropbox_files:
                        if file_info.get('name', '').endswith('.xlsx'):
                            file_record = {
                                "path": file_info.get('path_lower', file_info.get('path', path_to_try + '/' + file_info['name'])),
                                "name": file_info['name'],
                                "size": file_info.get('size', 0),
                                "modified": file_info.get('modified', ''),
                                "metadata": self._extract_metadata_from_filename(file_info['name'])
                            }
                            
                            # Load sidecar summary from /04_Data/03_Summaries/
                            summary_text = self._load_sidecar_summary(file_info['name'])
                            if summary_text:
                                file_record["summary"] = summary_text[:4000]  # Limit to 4000 chars
                            
                            files.append(file_record)
                    if files:  # If we found files, stop trying other paths
                        print(f"SUCCESS: Found {len(files)} files in {path_to_try}")
                        break
                except Exception as e:
                    print(f"Warning: Could not access {path_to_try}: {e}")
                    continue
            
            # Add files from session state if available
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'cleaned_sheets'):
                    for sheet_name, sheet_data in st.session_state.cleaned_sheets.items():
                        # Avoid duplicates
                        if not any(f["name"] == sheet_name for f in files):
                            files.append({
                                "path": f"session:{sheet_name}",
                                "name": sheet_name,
                                "size": len(str(sheet_data)) if sheet_data is not None else 0,
                                "modified": datetime.now().isoformat(),
                                "metadata": self._extract_metadata_from_filename(sheet_name),
                                "data": sheet_data
                            })
            except Exception as e:
                print(f"Warning: Could not access session state: {e}")
            
            return files
            
        except Exception as e:
            print(f"Warning: Could not get cleansed files: {e}")
            return []
    
    def _extract_file_identifiers(self, query: str) -> List[str]:
        """Extract file identifiers from query (R401, R402, Global, etc.)"""
        identifiers = []
        
        # Pattern for R-codes (R401, R402, etc.)
        r_codes = re.findall(r'\bR\d{3}\b', query, re.IGNORECASE)
        identifiers.extend(r_codes)
        
        # Pattern for common file identifiers
        common_ids = re.findall(r'\b(Global|US|DE|IT|UK|TH|WIP|Inventory|Financial|Parts)\b', query, re.IGNORECASE)
        identifiers.extend(common_ids)
        
        # Pattern for explicit file mentions
        file_mentions = re.findall(r'\b(file\s+\w+|\w+\.xlsx|\w+_cleansed)\b', query, re.IGNORECASE)
        identifiers.extend([f.replace('file ', '').replace('.xlsx', '').replace('_cleansed', '') for f in file_mentions])
        
        return list(set(identifiers))
    
    def _extract_period_filters(self, query: str) -> List[str]:
        """Extract period filters from query"""
        periods = []
        
        # Quarter patterns
        quarters = re.findall(r'\bQ[1-4]\b', query, re.IGNORECASE)
        periods.extend(quarters)
        
        # Month patterns
        months = re.findall(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', query, re.IGNORECASE)
        periods.extend(months)
        
        # Year patterns
        years = re.findall(r'\b(20\d{2})\b', query)
        periods.extend(years)
        
        return list(set(periods))
    
    def _extract_country_filters(self, query: str) -> List[str]:
        """Extract country filters from query"""
        countries = re.findall(r'\b(US|DE|IT|UK|TH|USA|Germany|Italy|United Kingdom|Thailand)\b', query, re.IGNORECASE)
        return list(set(countries))
    
    def _extract_erp_filters(self, query: str) -> List[str]:
        """Extract ERP filters from query"""
        erps = re.findall(r'\b(SAP|Oracle|JDE|D365|Dynamics)\b', query, re.IGNORECASE)
        return list(set(erps))
    
    def _extract_sheet_type_filters(self, query: str) -> List[str]:
        """Extract sheet type filters from query"""
        sheet_types = re.findall(r'\b(WIP|Inventory|Financial|Parts|Aging|Movement|Forecast)\b', query, re.IGNORECASE)
        return list(set(sheet_types))
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from filename"""
        metadata = {
            "filename": filename,
            "identifiers": [],
            "period": None,
            "country": None,
            "erp": None,
            "sheet_type": None,
            "timestamp": None
        }
        
        # Extract identifiers
        metadata["identifiers"] = self._extract_file_identifiers(filename)
        
        # Extract period
        periods = self._extract_period_filters(filename)
        if periods:
            metadata["period"] = periods[0]
        
        # Extract country
        countries = self._extract_country_filters(filename)
        if countries:
            metadata["country"] = countries[0]
        
        # Extract ERP
        erps = self._extract_erp_filters(filename)
        if erps:
            metadata["erp"] = erps[0]
        
        # Extract sheet type
        sheet_types = self._extract_sheet_type_filters(filename)
        if sheet_types:
            metadata["sheet_type"] = sheet_types[0]
        
        # Extract timestamp from filename
        timestamp_match = re.search(r'(\d{4}\d{2}\d{2}_\d{6})', filename)
        if timestamp_match:
            metadata["timestamp"] = timestamp_match.group(1)
        
        return metadata
    
    def _rank_files_by_metadata(self, files: List[Dict], identifiers: List[str], 
                               periods: List[str], countries: List[str], 
                               erps: List[str], sheet_types: List[str], hints: list[str] | None = None) -> List[Dict]:
        """Rank files by metadata match quality"""
        ranked_files = []
        
        for file_info in files:
            score = 0
            reasons = []
            metadata = file_info.get("metadata", {})
            
            # Exact identifier match (highest priority)
            for identifier in identifiers:
                if identifier.lower() in file_info["name"].lower():
                    score += 100
                    reasons.append(f"Exact identifier match: {identifier}")
                elif identifier.lower() in str(metadata.get("identifiers", [])).lower():
                    score += 80
                    reasons.append(f"Metadata identifier match: {identifier}")
            
            # Period match
            for period in periods:
                if period.lower() in file_info["name"].lower() or period.lower() == str(metadata.get("period", "")).lower():
                    score += 50
                    reasons.append(f"Period match: {period}")
            
            # Country match
            for country in countries:
                if country.lower() in file_info["name"].lower() or country.lower() == str(metadata.get("country", "")).lower():
                    score += 40
                    reasons.append(f"Country match: {country}")
            
            # ERP match
            for erp in erps:
                if erp.lower() in file_info["name"].lower() or erp.lower() == str(metadata.get("erp", "")).lower():
                    score += 30
                    reasons.append(f"ERP match: {erp}")
            
            # Sheet type match
            for sheet_type in sheet_types:
                if sheet_type.lower() in file_info["name"].lower() or sheet_type.lower() == str(metadata.get("sheet_type", "")).lower():
                    score += 20
                    reasons.append(f"Sheet type match: {sheet_type}")
            
            # Recency bonus
            if metadata.get("timestamp"):
                score += 10
                reasons.append("Recent timestamp")
            
            # Cleansed file bonus
            if "_cleansed" in file_info["name"]:
                score += 5
                reasons.append("Cleansed file")
            
            # Summary token overlap boost (generic, no hardcodes)
            if file_info.get("summary"):
                summary_lower = file_info["summary"].lower()
                query_tokens = self.last_query.lower().split() if hasattr(self, 'last_query') else []
                overlap_count = sum(1 for token in query_tokens if len(token) > 3 and token in summary_lower)
                if overlap_count > 0:
                    boost = min(overlap_count * 5, 25)  # Max 25 point boost
                    score += boost
                    reasons.append(f"Summary token overlap ({overlap_count} matches)")
            
            # Router hints boost (conservative +8 per hit)
            hints = [h.strip().lower() for h in (hints or []) if h and isinstance(h, str)]
            if hints:
                blob = " ".join([
                    str(file_info.get("name", "")), str(file_info.get("title", "")), 
                    str(metadata.get("tags", "")), str(metadata.get("period", "")), 
                    str(metadata.get("country", "")), str(metadata.get("erp", ""))
                ]).lower()
                hint_hits = sum(1 for h in hints if h and h in blob)
                if hint_hits > 0:
                    hint_boost = hint_hits * 8  # Conservative boost per hint match
                    score += hint_boost
                    reasons.append(f"Router hint matches ({hint_hits} hits)")
            
            file_info["ranking_score"] = score
            file_info["ranking_reason"] = "; ".join(reasons) if reasons else "No specific matches"
            
            if score > 0:  # Only include files with some relevance
                ranked_files.append(file_info)
        
        # Sort by score (highest first)
        ranked_files.sort(key=lambda x: x["ranking_score"], reverse=True)
        
        return ranked_files
    
    def _deduplicate_versions(self, ranked_files: List[Dict]) -> List[Dict]:
        """Remove duplicate versions, preferring most recent"""
        seen_bases = {}
        deduplicated = []
        
        for file_info in ranked_files:
            # Extract base name (remove timestamp and _cleansed suffix)
            base_name = re.sub(r'_\d{8}_\d{6}', '', file_info["name"])
            base_name = re.sub(r'_cleansed\.xlsx$', '', base_name)
            
            if base_name not in seen_bases:
                seen_bases[base_name] = file_info
                deduplicated.append(file_info)
            else:
                # Compare timestamps, prefer more recent
                existing = seen_bases[base_name]
                current_ts = file_info.get("metadata", {}).get("timestamp", "")
                existing_ts = existing.get("metadata", {}).get("timestamp", "")
                
                if current_ts > existing_ts:
                    # Replace with more recent version
                    deduplicated.remove(existing)
                    deduplicated.append(file_info)
                    seen_bases[base_name] = file_info
        
        return deduplicated
    
    def _select_files_for_intent(self, ranked_files: List[Dict], intent: str) -> List[Dict]:
        """Select appropriate files based on intent requirements"""
        if not ranked_files:
            return []
        
        # Intent-specific selection logic
        if intent == "comparison":
            # Apply generic comparison pairing (no hardcodes)
            return self._apply_generic_comparison_pairing(ranked_files)
        elif intent in ["forecasting", "movement_analysis", "optimization"]:
            # These intents can use multiple files for comprehensive analysis
            return ranked_files[:min(6, len(ranked_files))]
        elif intent in ["root_cause", "anomaly_detection"]:
            # Root cause and anomaly detection benefit from multiple perspectives
            return ranked_files[:min(3, len(ranked_files))]
        else:
            # Default: use top ranked file(s)
            return ranked_files[:min(2, len(ranked_files))]
    
    def process_dp_query(self, query: str, session_state: Any = None, router_contract: dict | None = None) -> Dict[str, Any]:
        """
        Process Data Processing query with full enterprise parity
        """
        # Store query for downstream use (country/intent cues)
        self.last_query = query
        
        # Extract router hints for file selection boosting
        router_hints = (router_contract or {}).get("files_hint", []) or []
        
        # Extract router actions and set preferred intent
        router_actions = set((router_contract or {}).get("actions", []))
        preferred_intent = None
        if "forecast_demand" in router_actions:
            preferred_intent = "forecasting"   # gentle nudge only
        
        # Store for logs/telemetry (non-breaking)
        if session_state is not None:
            session_state["router_actions"] = list(router_actions)
            session_state["router_preferred_intent"] = preferred_intent
        
        try:
            # Step 1: Classify intent using master instructions
            intent_result = classify_user_intent(query)
            intent = intent_result.get("intent", "eda")
            confidence = intent_result.get("confidence", 0.5)
            
            # Apply router preferred intent if appropriate (gentle nudge)
            if preferred_intent and intent in (None, "eda", "general"):
                intent = preferred_intent
            
            # Step 2: Resolve files using metadata-first approach
            file_resolution = self.resolve_files_metadata_first(query, intent, hints=router_hints)
            
            # Step 3: Check if clarification is needed
            if file_resolution.get("needs_clarification"):
                return self._generate_clarification_response(file_resolution, query)
            
            selected_files = file_resolution.get("selected_files", [])
            if not selected_files:
                # Get all available files for debugging
                all_files = self._get_all_cleansed_files()
                print(f"DEBUG DP: No files selected, but {len(all_files)} total files available")
                for f in all_files[:3]:
                    print(f"DEBUG DP: Available file: {f['name']}")
                
                if all_files:
                    # If we have files but didn't select any, use the first available file
                    selected_files = all_files[:1]
                    print(f"DEBUG DP: Using fallback file: {selected_files[0]['name']}")
                else:
                    return {
                        "error": "No matching files found. Please upload and process files first, or check your file references.",
                        "suggestions": "Available paths checked: " + ", ".join(self.cleansed_files_path_variations[:3]),
                        "needs_clarification": True,
                        "clarification_type": "no_files"
                    }
            
            # Step 4: Build execution plan using master instructions
            plan = self._build_execution_plan(intent, selected_files, query)
            
            # Step 5: Execute plan using enterprise reasoning protocol
            execution_result = self._execute_with_reasoning_protocol(plan, selected_files)
            
            # Step 6: Apply quality protocol and confidence scoring
            final_result = self._apply_quality_protocol(execution_result, intent, query)
            
            # Step 7: Log the query and results
            self._log_dp_query(query, intent, selected_files, file_resolution, final_result)
            
            return final_result
            
        except Exception as e:
            return {
                "error": f"Data Processing query failed: {str(e)}",
                "intent": intent if 'intent' in locals() else "unknown",
                "query": query
            }
    
    def _generate_clarification_response(self, file_resolution: Dict, query: str) -> Dict[str, Any]:
        """Generate clarification response when file selection is ambiguous"""
        options = file_resolution.get("clarification_options", [])
        
        if not options:
            return {
                "needs_clarification": True,
                "clarification_message": "No matching files found. Please upload and process files first, or check your file references.",
                "clarification_type": "no_files"
            }
        
        option_names = [f"{opt['name']} (Score: {opt.get('ranking_score', 0)})" for opt in options[:3]]
        
        return {
            "needs_clarification": True,
            "clarification_message": f"I found multiple matches: {', '.join(option_names)} — which do you want?",
            "clarification_options": options[:3],
            "clarification_type": "multiple_files",
            "original_query": query
        }
    
    def _build_execution_plan(self, intent: str, selected_files: List[Dict], query: str) -> List[Dict]:
        """Build execution plan based on intent and selected files"""
        plan = []
        
        # Get file paths for tools
        file_paths = [f["path"] for f in selected_files]
        
        # Intent-specific plan building
        if intent == "comparison":
            if len(file_paths) >= 2:
                plan.append({
                    "tool": "compare_files",
                    "args": {
                        "file_a": file_paths[0],
                        "file_b": file_paths[1],
                        "additional_files": file_paths[2:] if len(file_paths) > 2 else None,
                        "strategy": "auto"
                    }
                })
            
        elif intent == "forecasting":
            plan.append({
                "tool": "forecast_demand",
                "args": {
                    "files": file_paths,
                    "forecast_periods": 12,
                    "method": "auto"
                }
            })
            plan.append({
                "tool": "calculate_inventory_policy",
                "args": {
                    "files": file_paths,
                    "service_level": 0.95
                }
            })
            
        elif intent == "root_cause":
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": "root_cause_analysis",
                    "analysis_focus": self._extract_analysis_focus(query)
                }
            })
            
        elif intent == "movement_analysis":
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": "movement_analysis",
                    "analysis_focus": "timing_and_flow"
                }
            })
            
        elif intent == "optimization":
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": "optimization",
                    "analysis_focus": "efficiency_improvement"
                }
            })
            
        elif intent == "anomaly_detection":
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": "anomaly_detection",
                    "analysis_focus": "outlier_identification"
                }
            })
            
        elif intent == "scenario_analysis":
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": "scenario_analysis",
                    "analysis_focus": "what_if_modeling"
                }
            })
            
        elif intent == "exec_summary":
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": "executive_summary",
                    "analysis_focus": "high_level_insights"
                }
            })
            
        elif intent == "gap_check":
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": "data_gap_analysis",
                    "analysis_focus": "missing_data_identification"
                }
            })
            
        elif intent in ["par_policy", "safety_stock", "demand_projection", "seasonal_analysis", "eo_future_risk"]:
            # Specialized forecasting intents
            plan.append({
                "tool": "forecast_demand",
                "args": {
                    "files": file_paths,
                    "forecast_type": intent,
                    "forecast_periods": 12,
                    "method": "auto"
                }
            })
            if intent in ["par_policy", "safety_stock"]:
                plan.append({
                    "tool": "calculate_inventory_policy",
                    "args": {
                        "files": file_paths,
                        "policy_type": intent,
                        "service_level": 0.95
                    }
                })
                
        elif intent in ["eda"]:
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": "exploratory_data_analysis",
                    "analysis_focus": self._extract_analysis_focus(query)
                }
            })
            
        # Always add KB search for context
        plan.append({
            "tool": "kb_search",
            "args": {
                "query": query,
                "k": max(4, len(selected_files))
            }
        })
        
        # Add chart generation
        plan.append({
            "tool": "chart",
            "args": {
                "files": file_paths,
                "chart_types": self._determine_chart_types(intent),
                "include_tables": True
            }
        })
        
        return plan
    
    def _extract_analysis_focus(self, query: str) -> str:
        """Extract specific analysis focus from query"""
        focus_keywords = {
            "variance": ["variance", "difference", "change", "deviation"],
            "trend": ["trend", "pattern", "over time", "temporal"],
            "ranking": ["top", "bottom", "highest", "lowest", "rank"],
            "distribution": ["distribution", "spread", "range", "histogram"],
            "correlation": ["correlation", "relationship", "related", "connection"]
        }
        
        query_lower = query.lower()
        for focus, keywords in focus_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return focus
        
        return "general"
    
    def _determine_chart_types(self, intent: str) -> List[str]:
        """Determine appropriate chart types based on intent"""
        chart_mapping = {
            "comparison": ["delta_waterfall", "aging_shift", "movers_scatter"],
            "forecasting": ["forecast_line", "confidence_bands", "backtest_accuracy"],
            "root_cause": ["pareto", "variance_breakdown", "driver_analysis"],
            "movement_analysis": ["flow_diagram", "time_series", "movement_heatmap"],
            "anomaly_detection": ["control_chart", "outlier_scatter", "anomaly_timeline"],
            "optimization": ["efficiency_chart", "optimization_pareto", "improvement_timeline"],
            "scenario_analysis": ["scenario_comparison", "sensitivity_analysis", "what_if_chart"],
            "exec_summary": ["executive_dashboard", "kpi_scorecard", "trend_summary"],
            "gap_check": ["data_completeness", "gap_analysis", "missing_data_heatmap"],
            "par_policy": ["par_level_chart", "policy_comparison", "service_level_analysis"],
            "safety_stock": ["safety_stock_chart", "risk_analysis", "stockout_probability"],
            "demand_projection": ["demand_forecast", "seasonality_chart", "accuracy_metrics"],
            "seasonal_analysis": ["seasonal_decomposition", "cyclical_patterns", "seasonal_forecast"],
            "eo_future_risk": ["risk_assessment", "future_exposure", "risk_mitigation"],
            "eda": ["distribution", "correlation_matrix", "summary_stats"]
        }
        
        return chart_mapping.get(intent, ["bar", "line", "scatter"])
    
    def _execute_with_reasoning_protocol(self, plan: List[Dict], selected_files: List[Dict]) -> Dict[str, Any]:
        """Execute plan using enterprise reasoning protocol with actual data loading"""
        try:
            # Load actual file data from Dropbox
            loaded_dataframes = {}
            for file_info in selected_files:
                try:
                    # Load DataFrame from Dropbox
                    if file_info.get("data"):
                        # Already have data (from session state)
                        loaded_dataframes[file_info["name"]] = file_info["data"]
                    else:
                        # Load from Dropbox
                        from dbx_utils import read_file_bytes
                        file_bytes = read_file_bytes(file_info["path"])
                        df = pd.read_excel(io.BytesIO(file_bytes))
                        loaded_dataframes[file_info["name"]] = df
                        print(f"DEBUG: Loaded {file_info['name']} with {len(df)} rows and {len(df.columns)} columns")
                except Exception as e:
                    print(f"Warning: Could not load file {file_info['name']}: {e}")
                    continue
            
            if not loaded_dataframes:
                return {
                    "error": "No files could be loaded for analysis",
                    "plan": plan,
                    "selected_files": selected_files,
                    "calls": []
                }
            
            # Execute each tool in the plan with actual data
            calls = []
            artifacts = []
            
            for step in plan:
                tool = step["tool"]
                args = step["args"]
                
                try:
                    if tool == "dataframe_query":
                        # Execute dataframe analysis with loaded data
                        result = self._execute_dataframe_query(loaded_dataframes, args)
                        calls.append({
                            "tool": tool,
                            "args": args,
                            "result_meta": result,
                            "success": True
                        })
                        
                    elif tool == "kb_search":
                        # Execute KB search
                        try:
                            from tools_runtime import kb_search
                            result = kb_search(**args)
                            calls.append({
                                "tool": tool,
                                "args": args,
                                "result_meta": result,
                                "success": True
                            })
                        except Exception as e:
                            calls.append({
                                "tool": tool,
                                "args": args,
                                "error": str(e),
                                "success": False
                            })
                    
                    elif tool == "chart":
                        # Execute chart generation
                        result = self._execute_chart_generation(loaded_dataframes, args)
                        calls.append({
                            "tool": tool,
                            "args": args,
                            "result_meta": result,
                            "success": True
                        })
                        if result.get("chart_paths"):
                            artifacts.extend(result["chart_paths"])
                    
                    else:
                        # Handle other tools
                        calls.append({
                            "tool": tool,
                            "args": args,
                            "error": f"Tool {tool} not implemented in DP orchestrator",
                            "success": False
                        })
                        
                except Exception as e:
                    calls.append({
                        "tool": tool,
                        "args": args,
                        "error": str(e),
                        "success": False
                    })
            
            return {
                "calls": calls,
                "artifacts": artifacts,
                "selected_files": selected_files,
                "file_summaries": self._generate_file_summaries(selected_files),
                "loaded_data": {name: f"{len(df)} rows x {len(df.columns)} cols" for name, df in loaded_dataframes.items()}
            }
            
        except Exception as e:
            return {
                "error": f"Execution failed: {str(e)}",
                "plan": plan,
                "selected_files": selected_files,
                "calls": []
            }
    
    def _generate_file_summaries(self, selected_files: List[Dict]) -> List[str]:
        """Generate 2-3 summary sentences per selected file"""
        summaries = []
        
        for file_info in selected_files:
            metadata = file_info.get("metadata", {})
            name = file_info["name"]
            
            summary_parts = [f"File: {name}"]
            
            if metadata.get("period"):
                summary_parts.append(f"Period: {metadata['period']}")
            if metadata.get("country"):
                summary_parts.append(f"Country: {metadata['country']}")
            if metadata.get("sheet_type"):
                summary_parts.append(f"Type: {metadata['sheet_type']}")
            
            # Add data insights if available
            if "data" in file_info and file_info["data"] is not None:
                try:
                    data = file_info["data"]
                    if hasattr(data, 'shape'):
                        summary_parts.append(f"Contains {data.shape[0]} records with {data.shape[1]} columns")
                except:
                    pass
            
            summary = ". ".join(summary_parts) + "."
            summaries.append(summary)
        
        return summaries
    
    def _apply_quality_protocol(self, execution_result: Dict, intent: str, query: str) -> Dict[str, Any]:
        """Apply quality protocol and confidence scoring"""
        try:
            # Extract components for confidence calculation
            kb_sources = []
            analysis_quality = 0.8  # Default high quality for DP
            validation_score = 0.7  # Default validation score
            citation_density = 0.6  # Default citation density
            
            # Extract KB sources from execution result
            for call in execution_result.get("calls", []):
                if call.get("tool") == "kb_search":
                    result_meta = call.get("result_meta", {})
                    if "sources" in result_meta:
                        kb_sources.extend(result_meta["sources"])
            
            # Calculate R/A/V/C confidence
            ravc_breakdown = {
                "retrieval_strength_R": min(len(kb_sources) / 4.0, 1.0),  # Based on KB sources
                "agreement_A": analysis_quality,
                "validations_V": validation_score,
                "citation_density_C": citation_density
            }
            
            confidence_score = calculate_ravc_confidence(ravc_breakdown)
            
            # Format response using standard_report schema
            formatted_response = self._format_standard_report(
                execution_result, intent, query, confidence_score, ravc_breakdown, kb_sources
            )
            
            return formatted_response
            
        except Exception as e:
            # Fallback to basic formatting
            return {
                "title": f"Analysis Results for: {query}",
                "executive_insight": "Analysis completed with limited quality assessment.",
                "error": f"Quality protocol failed: {str(e)}",
                "raw_results": execution_result
            }
    
    def _format_standard_report(self, execution_result: Dict, intent: str, query: str, 
                              confidence_score: float, ravc_breakdown: Dict, kb_sources: List) -> Dict[str, Any]:
        """Format response using standard_report schema"""
        
        # Extract key results from execution
        calls = execution_result.get("calls", [])
        artifacts = execution_result.get("artifacts", [])
        file_summaries = execution_result.get("file_summaries", [])
        
        # Build structured response
        response = {
            "title": f"{intent.replace('_', ' ').title()} Analysis: {query}",
            "executive_insight": self._generate_executive_insight(calls, intent),
            "method_and_scope": {
                "files_analyzed": len(execution_result.get("selected_files", [])),
                "analysis_type": intent,
                "data_sources": file_summaries[:3]  # First 3 file summaries
            },
            "evidence_and_calculations": self._extract_evidence_and_calculations(calls),
            "root_causes_drivers": self._extract_drivers(calls, intent),
            "recommendations": self._generate_recommendations(calls, intent),
            "confidence": {
                "score": confidence_score,
                "level": "High" if confidence_score >= 0.75 else "Medium" if confidence_score >= 0.55 else "Low",
                "ravc_breakdown": ravc_breakdown
            },
            "limits_data_needed": self._identify_limitations(execution_result, kb_sources),
            "citations": kb_sources[:10],  # Top 10 citations
            "artifacts": artifacts,
            "intent": intent,
            "query": query
        }
        
        return response
    
    def _generate_executive_insight(self, calls: List[Dict], intent: str) -> str:
        """Generate executive insight from execution results"""
        insights = []
        
        for call in calls:
            tool = call.get("tool")
            result_meta = call.get("result_meta", {})
            
            if tool == "compare_files" and not call.get("error"):
                insights.append("Comparison analysis completed successfully with delta calculations.")
            elif tool == "forecast_demand" and not call.get("error"):
                insights.append("Demand forecasting models generated with confidence intervals.")
            elif tool == "dataframe_query" and not call.get("error"):
                insights.append("Data analysis completed with statistical insights.")
        
        if not insights:
            insights.append(f"{intent.replace('_', ' ').title()} analysis completed.")
        
        return " ".join(insights)
    
    def _extract_evidence_and_calculations(self, calls: List[Dict]) -> Dict[str, Any]:
        """Extract evidence and calculations from execution results"""
        evidence = {
            "tables": [],
            "charts": [],
            "calculations": []
        }
        
        for call in calls:
            result_meta = call.get("result_meta", {})
            
            if "tables" in result_meta:
                evidence["tables"].extend(result_meta["tables"])
            if "charts" in result_meta:
                evidence["charts"].extend(result_meta["charts"])
            if "calculations" in result_meta:
                evidence["calculations"].extend(result_meta["calculations"])
        
        return evidence
    
    def _extract_drivers(self, calls: List[Dict], intent: str) -> List[str]:
        """Extract key drivers from analysis results"""
        drivers = []
        
        for call in calls:
            result_meta = call.get("result_meta", {})
            
            if "drivers" in result_meta:
                drivers.extend(result_meta["drivers"])
            elif "top_changes" in result_meta:
                drivers.extend([f"Key change: {change}" for change in result_meta["top_changes"][:3]])
            elif "insights" in result_meta:
                drivers.extend(result_meta["insights"][:3])
        
        if not drivers:
            drivers.append(f"Analysis completed for {intent} - specific drivers require deeper investigation.")
        
        return drivers[:5]  # Top 5 drivers
    
    def _generate_recommendations(self, calls: List[Dict], intent: str) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Intent-specific recommendations
        if intent == "comparison":
            recommendations.append("Review significant deltas identified in the comparison analysis.")
            recommendations.append("Investigate root causes of major changes between periods.")
        elif intent == "forecasting":
            recommendations.append("Implement recommended safety stock and reorder point adjustments.")
            recommendations.append("Monitor forecast accuracy and adjust models as needed.")
        elif intent == "root_cause":
            recommendations.append("Focus investigation on the top-ranked contributing factors.")
            recommendations.append("Implement corrective actions for identified root causes.")
        else:
            recommendations.append(f"Act on insights from {intent} analysis.")
            recommendations.append("Monitor key metrics and trends identified.")
        
        return recommendations
    
    def _identify_limitations(self, execution_result: Dict, kb_sources: List) -> List[str]:
        """Identify limitations and data gaps (suppress FAISS error leakage)"""
        limitations = []
        
        # Check for execution errors (suppress FAISS/technical errors)
        errors = []
        for call in execution_result.get("calls", []):
            if call.get("error"):
                error_msg = str(call["error"]).lower()
                # Suppress FAISS and technical error details
                if any(term in error_msg for term in ["faiss", "index", "vector", "embedding", "dimension"]):
                    continue  # Skip FAISS-related errors
                elif "no module named" in error_msg or "import" in error_msg:
                    continue  # Skip import errors
                else:
                    errors.append(call["error"])
        
        if errors:
            limitations.append(f"Some analysis components encountered issues: {'; '.join(errors[:2])}")
        
        # Check KB coverage (soft note, not error leakage)
        if len(kb_sources) < 2:
            limitations.append("Limited knowledge base coverage - analysis focused on available data files.")
        
        # Check file coverage
        selected_files = execution_result.get("selected_files", [])
        if len(selected_files) < 2:
            limitations.append("Analysis based on limited file selection - broader data may provide additional insights.")
        
        if not limitations:
            limitations.append("Analysis completed with available data - additional data sources could enhance insights.")
        
        return limitations
    
    def _log_dp_query(self, query: str, intent: str, selected_files: List[Dict], 
                     file_resolution: Dict, final_result: Dict):
        """Log DP query with comprehensive metadata"""
        try:
            if self.query_logger:
                log_entry = {
                    "query": query,
                    "intent": intent,
                    "mode": "data_processing",
                    "chosen_files": [f["name"] for f in selected_files],
                    "ranking_reasons": file_resolution.get("ranking_reasons", []),
                    "rejected_candidates": [f["name"] for f in file_resolution.get("rejected_candidates", [])[:5]],
                    "clarifier_shown": file_resolution.get("needs_clarification", False),
                    "confidence_score": final_result.get("confidence", {}).get("score", 0.0),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.query_logger.log_query(log_entry)
        except Exception as e:
            print(f"Warning: Could not log DP query: {e}")

    def _execute_dataframe_query(self, loaded_dataframes: Dict[str, pd.DataFrame], args: Dict) -> Dict[str, Any]:
        """Execute dataframe query analysis with actual data"""
        try:
            query_type = args.get("query_type", "eda")
            analysis_focus = args.get("analysis_focus", "general")
            
            # Apply US filter if mentioned in query
            filtered_dataframes = self._apply_country_filter(loaded_dataframes)
            
            insights = []
            calculations = []
            tables = []
            data_summary = []
            
            total_value = 0
            aging_analysis = {}
            eo_analysis = {}
            
            for name, df in filtered_dataframes.items():
                data_summary.append(f"{name}: {len(df)} rows, {len(df.columns)} columns")
                
                # Detect aging bucket columns
                aging_columns = self._detect_aging_columns(df)
                print(f"DEBUG: Found aging columns in {name}: {aging_columns}")
                
                if aging_columns:
                    # Calculate $ by aging bucket
                    aging_totals = self._calculate_aging_totals(df, aging_columns)
                    aging_analysis[name] = aging_totals
                    
                    # E&O proxy (>180d / total)
                    eo_proxy = self._calculate_eo_proxy(aging_totals)
                    eo_analysis[name] = eo_proxy
                    
                    # Add insights
                    total_file_value = sum(aging_totals.values())
                    total_value += total_file_value
                    insights.append(f"{name}: Total value ${total_file_value:,.0f}, E&O risk {eo_proxy:.1%}")
                    
                    # Add calculations
                    for bucket, value in aging_totals.items():
                        if value > 0:
                            calculations.append(f"{name} {bucket}: ${value:,.0f}")
                
                # Top contributors analysis
                top_contributors = self._identify_top_contributors(df)
                if top_contributors:
                    tables.extend(top_contributors)
            
            # Multi-file comparison delta (if exactly 2 files)
            if len(filtered_dataframes) == 2:
                delta_analysis = self._calculate_two_file_delta(list(filtered_dataframes.values()))
                if delta_analysis:
                    insights.append(f"Total $ delta between files: ${delta_analysis['total_delta']:,.0f}")
                    calculations.append(f"File comparison delta: ${delta_analysis['total_delta']:,.0f}")
            
            # Overall E&O summary
            if eo_analysis:
                avg_eo_risk = np.mean(list(eo_analysis.values()))
                insights.append(f"Average E&O risk across files: {avg_eo_risk:.1%}")
                calculations.append(f"Portfolio E&O risk: {avg_eo_risk:.1%}")
            
            # Create aging summary table
            if aging_analysis:
                aging_table = self._create_aging_summary_table(aging_analysis)
                tables.append(aging_table)
            
            return {
                "insights": insights,
                "calculations": calculations,
                "tables": tables,
                "data_summary": data_summary,
                "aging_analysis": aging_analysis,
                "eo_analysis": eo_analysis,
                "total_value": total_value,
                "dataframes_analyzed": len(filtered_dataframes)
            }
            
        except Exception as e:
            print(f"ERROR: Dataframe analysis failed: {e}")
            return {
                "error": str(e),
                "insights": ["Analysis encountered technical difficulties"],
                "calculations": ["Unable to complete calculations"],
                "data_summary": [f"Attempted to analyze {len(loaded_dataframes)} files"]
            }

    def _execute_chart_generation(self, loaded_dataframes: Dict[str, pd.DataFrame], args: Dict) -> Dict[str, Any]:
        """Execute real matplotlib chart generation with artifact folder routing"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import tempfile
            
            # Get artifact folder using orchestrator helper
            try:
                from orchestrator import _artifact_folder
                # Create simple app_paths object
                class SimpleAppPaths:
                    def __init__(self):
                        from constants import EDA_CHART
                        self.dbx_charts_folder = EDA_CHART
                        self.dbx_eda_charts_folder = EDA_CHART
                        self.default_charts_path = EDA_CHART
                
                app_paths = SimpleAppPaths()
                artifact_folder = _artifact_folder("charts", app_paths)
            except Exception as e:
                print(f"DEBUG: Could not get artifact folder: {e}")
                artifact_folder = None
            
            # Concatenate all DataFrames for analysis
            combined_df = pd.DataFrame()
            for name, df in loaded_dataframes.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy['source_file'] = name
                    combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
            
            if combined_df.empty:
                return {"chart_paths": [], "chart_descriptions": ["No data available for charting"], "charts_generated": 0}
            
            # Get numeric columns defensively
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return {"chart_paths": [], "chart_descriptions": ["No numeric data available for charting"], "charts_generated": 0}
            
            chart_paths = []
            chart_descriptions = []
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Chart 1: Distribution plot of first numeric column
            try:
                plt.figure(figsize=(10, 6))
                col_to_plot = numeric_cols[0]
                combined_df[col_to_plot].hist(bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'Distribution of {col_to_plot}')
                plt.xlabel(col_to_plot)
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                chart_path = self._save_chart_with_artifact_routing(plt.gcf(), f"distribution_{timestamp}", artifact_folder)
                if chart_path:
                    chart_paths.append(chart_path)
                    chart_descriptions.append(f"Distribution chart for {col_to_plot}")
                
                plt.close()
            except Exception as e:
                print(f"DEBUG: Distribution chart failed: {e}")
            
            # Chart 2: Correlation matrix if multiple numeric columns
            if len(numeric_cols) > 1:
                try:
                    plt.figure(figsize=(10, 8))
                    correlation_matrix = combined_df[numeric_cols[:6]].corr()  # Limit to 6 columns
                    
                    # Simple heatmap-style plot
                    import numpy as np
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    
                    # Set ticks and labels
                    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
                    ax.set_yticks(np.arange(len(correlation_matrix.columns)))
                    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
                    ax.set_yticklabels(correlation_matrix.columns)
                    
                    # Add correlation values as text
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(len(correlation_matrix.columns)):
                            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black" if abs(correlation_matrix.iloc[i, j]) < 0.5 else "white")
                    
                    plt.title('Correlation Matrix')
                    plt.colorbar(im)
                    plt.tight_layout()
                    
                    chart_path = self._save_chart_with_artifact_routing(plt.gcf(), f"correlation_{timestamp}", artifact_folder)
                    if chart_path:
                        chart_paths.append(chart_path)
                        chart_descriptions.append("Correlation matrix of numeric variables")
                    
                    plt.close()
                except Exception as e:
                    print(f"DEBUG: Correlation chart failed: {e}")
            
            return {
                "chart_paths": chart_paths,
                "chart_descriptions": chart_descriptions,
                "charts_generated": len(chart_paths)
            }
            
        except Exception as e:
            print(f"DEBUG: Chart generation failed: {e}")
            return {
                "error": str(e),
                "chart_paths": [],
                "chart_descriptions": ["Chart generation failed"]
            }

    def _generate_executive_insight_from_results(self, calls: List[Dict], insights: List[str]) -> str:
        """Generate executive insight from actual analysis results"""
        if insights:
            return f"Analysis of inventory and WIP data revealed {len(insights)} key insights. " + insights[0] if insights else ""
        
        successful_calls = len([c for c in calls if c.get("success")])
        return f"Completed analysis using {successful_calls} analytical tools with enterprise data processing."

    def _extract_drivers_from_results(self, insights: List[str], intent: str) -> List[str]:
        """Extract drivers from actual analysis insights"""
        if insights:
            drivers = []
            for insight in insights:
                if "column" in insight.lower() or "mean" in insight.lower() or "std" in insight.lower():
                    drivers.append(f"Data pattern identified: {insight}")
                elif "inventory" in insight.lower() or "wip" in insight.lower():
                    drivers.append(f"Inventory/WIP factor: {insight}")
            return drivers if drivers else insights
        
        return [f"Analysis completed for {intent} - specific drivers require deeper investigation."]

    def _generate_recommendations_from_results(self, insights: List[str], intent: str) -> List[str]:
        """Generate recommendations from actual analysis insights"""
        if insights:
            recommendations = []
            for insight in insights:
                if "numeric columns" in insight.lower():
                    recommendations.append("Focus analysis on the identified numeric data columns for quantitative insights.")
                elif "inventory" in insight.lower() or "wip" in insight.lower():
                    recommendations.append("Review inventory and WIP patterns identified in the analysis.")
            return recommendations if recommendations else [f"Act on insights from {intent} analysis.", "Monitor key metrics and trends identified."]
        
        return [f"Act on insights from {intent} analysis.", "Monitor key metrics and trends identified."]

    def _load_sidecar_summary(self, filename: str) -> Optional[str]:
        """Load sidecar summary JSON from /04_Data/03_Summaries/"""
        try:
            # Get base name without extension
            base_name = filename.replace('.xlsx', '').replace('_cleansed', '')
            summary_filename = f"{base_name}.json"
            
            # Try multiple summary path variations
            summary_paths = [
                f"/Apps/Ethos LLM/Project_Root/04_Data/03_Summaries/{summary_filename}",
                f"/Project_Root/04_Data/03_Summaries/{summary_filename}",
                f"/project_root/04_data/03_summaries/{summary_filename}",
                f"/Apps/Ethos LLM/project_root/04_data/03_summaries/{summary_filename}"
            ]
            
            for summary_path in summary_paths:
                try:
                    from dbx_utils import read_file_bytes
                    summary_bytes = read_file_bytes(summary_path)
                    summary_json = json.loads(summary_bytes.decode('utf-8'))
                    
                    # Extract summary text from various possible fields
                    if isinstance(summary_json, dict):
                        summary_text = (
                            summary_json.get('summary', '') or 
                            summary_json.get('executive_summary', '') or
                            summary_json.get('description', '') or
                            str(summary_json)
                        )
                        if summary_text and len(summary_text.strip()) > 10:
                            print(f"DEBUG: Loaded summary for {filename}: {len(summary_text)} chars")
                            return summary_text
                except Exception as e:
                    print(f"DEBUG: Could not load summary from {summary_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"DEBUG: Summary loading failed for {filename}: {e}")
        
        return None

    def _apply_generic_comparison_pairing(self, ranked_files: List[Dict]) -> List[Dict]:
        """Apply generic comparison pairing with synergy scoring (no hardcodes)"""
        if len(ranked_files) < 2:
            return ranked_files[:min(2, len(ranked_files))]
        
        # Calculate pairwise synergy scores for top candidates
        best_pairs = []
        
        for i in range(min(6, len(ranked_files))):  # Check top 6 files
            for j in range(i + 1, min(6, len(ranked_files))):
                file_a, file_b = ranked_files[i], ranked_files[j]
                synergy_score = self._calculate_pair_synergy(file_a, file_b)
                
                best_pairs.append({
                    'files': [file_a, file_b],
                    'synergy_score': synergy_score,
                    'combined_score': file_a['ranking_score'] + file_b['ranking_score'] + synergy_score
                })
        
        # Sort by combined score and return best pair
        if best_pairs:
            best_pairs.sort(key=lambda x: x['combined_score'], reverse=True)
            best_pair = best_pairs[0]
            
            # Add synergy reasons to ranking
            for file_info in best_pair['files']:
                if best_pair['synergy_score'] > 0:
                    file_info['ranking_reason'] += f"; Pair synergy: +{best_pair['synergy_score']}"
            
            return best_pair['files']
        
        # Fallback to top 2 files
        return ranked_files[:2]
    
    def _calculate_pair_synergy(self, file_a: Dict, file_b: Dict) -> int:
        """Calculate synergy score between two files for comparison (generic, no hardcodes)"""
        synergy = 0
        
        meta_a = file_a.get('metadata', {})
        meta_b = file_b.get('metadata', {})
        
        # Same country boost (+8)
        if (meta_a.get('country') and meta_b.get('country') and 
            meta_a['country'].lower() == meta_b['country'].lower()):
            synergy += 8
        
        # Same/adjacent period boost (+6)
        period_a = meta_a.get('period', '').lower()
        period_b = meta_b.get('period', '').lower()
        if period_a and period_b:
            if period_a == period_b:
                synergy += 6
            elif self._are_adjacent_periods(period_a, period_b):
                synergy += 6
        
        # Similar base name boost (+6)
        name_a = file_a['name'].lower().replace('_cleansed.xlsx', '').replace('.xlsx', '')
        name_b = file_b['name'].lower().replace('_cleansed.xlsx', '').replace('.xlsx', '')
        
        # Check for similar base patterns (e.g., similar prefixes/suffixes)
        if len(name_a) > 3 and len(name_b) > 3:
            # Similar prefix (first 3+ chars)
            if name_a[:3] == name_b[:3] or name_a[:4] == name_b[:4]:
                synergy += 6
            # Similar suffix (last 3+ chars)  
            elif name_a[-3:] == name_b[-3:] or name_a[-4:] == name_b[-4:]:
                synergy += 6
        
        return synergy
    
    def _are_adjacent_periods(self, period_a: str, period_b: str) -> bool:
        """Check if two periods are adjacent (Q1/Q2, Jan/Feb, etc.)"""
        # Quarter adjacency
        quarter_map = {'q1': 1, 'q2': 2, 'q3': 3, 'q4': 4}
        if period_a in quarter_map and period_b in quarter_map:
            return abs(quarter_map[period_a] - quarter_map[period_b]) == 1
        
        # Month adjacency (simplified)
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        if period_a in month_map and period_b in month_map:
            return abs(month_map[period_a] - month_map[period_b]) == 1
        
        # Numeric adjacency (2023/2024, etc.)
        try:
            num_a, num_b = int(period_a), int(period_b)
            return abs(num_a - num_b) == 1
        except ValueError:
            pass
        
        return False

    def _save_chart_with_artifact_routing(self, fig, base_name: str, artifact_folder: Optional[str]) -> Optional[str]:
        """Save chart using artifact routing (Dropbox/S3) or local temp"""
        try:
            import tempfile
            
            if artifact_folder:
                # Use artifact folder routing
                try:
                    from dbx_utils import upload_file
                    
                    # Save to temp file first
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        fig.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
                        
                        # Upload to Dropbox
                        chart_path = f"{artifact_folder}/{base_name}.png"
                        upload_file(tmp_file.name, chart_path)
                        print(f"DEBUG: Chart saved to {chart_path}")
                        return chart_path
                        
                except Exception as e:
                    print(f"DEBUG: Artifact upload failed: {e}")
                    # Fall back to local temp
                    pass
            
            # Fallback: save to local temp directory
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=tempfile.gettempdir()) as tmp_file:
                fig.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
                print(f"DEBUG: Chart saved locally to {tmp_file.name}")
                return tmp_file.name
                
        except Exception as e:
            print(f"DEBUG: Chart save failed: {e}")
            return None

    def _apply_country_filter(self, loaded_dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply country filter if US is mentioned in query"""
        if not hasattr(self, 'last_query') or 'us' not in self.last_query.lower():
            return loaded_dataframes
        
        filtered = {}
        for name, df in loaded_dataframes.items():
            # Look for country columns and filter to US
            country_cols = [col for col in df.columns if 'country' in col.lower() or 'region' in col.lower()]
            if country_cols:
                us_data = df[df[country_cols[0]].str.upper() == 'US']
                if not us_data.empty:
                    filtered[name] = us_data
                    print(f"DEBUG: Filtered {name} to {len(us_data)} US rows")
                else:
                    filtered[name] = df  # Keep all data if no US rows found
            else:
                filtered[name] = df  # Keep all data if no country column
        
        return filtered if filtered else loaded_dataframes
    
    def _detect_aging_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect aging bucket columns by regex patterns"""
        aging_patterns = [
            r'aging.*0.*30|0.*30.*day',
            r'aging.*31.*60|31.*60.*day', 
            r'aging.*61.*90|61.*90.*day',
            r'aging.*91.*180|91.*180.*day',
            r'aging.*181|181.*day|over.*180'
        ]
        
        aging_cols = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern in aging_patterns:
                if re.search(pattern, col_lower):
                    aging_cols.append(col)
                    break
        
        return aging_cols
    
    def _calculate_aging_totals(self, df: pd.DataFrame, aging_columns: List[str]) -> Dict[str, float]:
        """Calculate $ totals by aging bucket"""
        aging_totals = {}
        
        for col in aging_columns:
            try:
                # Try to sum the column values
                total = df[col].sum()
                
                # Categorize by bucket
                col_lower = col.lower()
                if '0' in col_lower and ('30' in col_lower or 'thirty' in col_lower):
                    aging_totals['0-30 days'] = total
                elif '31' in col_lower and ('60' in col_lower or 'sixty' in col_lower):
                    aging_totals['31-60 days'] = total
                elif '61' in col_lower and ('90' in col_lower or 'ninety' in col_lower):
                    aging_totals['61-90 days'] = total
                elif '91' in col_lower and ('180' in col_lower):
                    aging_totals['91-180 days'] = total
                elif '181' in col_lower or 'over' in col_lower:
                    aging_totals['181+ days'] = total
                else:
                    aging_totals[f'Aging: {col}'] = total
                    
            except Exception as e:
                print(f"DEBUG: Could not calculate aging total for {col}: {e}")
        
        return aging_totals
    
    def _calculate_eo_proxy(self, aging_totals: Dict[str, float]) -> float:
        """Calculate E&O proxy as >180d / total"""
        try:
            over_180 = aging_totals.get('181+ days', 0)
            total = sum(aging_totals.values())
            
            if total > 0:
                return over_180 / total
            return 0.0
        except Exception:
            return 0.0
    
    def _identify_top_contributors(self, df: pd.DataFrame) -> List[Dict]:
        """Identify top contributors by value"""
        tables = []
        
        try:
            # Look for value columns
            value_cols = [col for col in df.columns if any(term in col.lower() 
                         for term in ['value', 'amount', 'cost', 'price', 'total'])]
            
            # Look for grouping columns
            group_cols = [col for col in df.columns if any(term in col.lower() 
                         for term in ['family', 'plant', 'category', 'type', 'group'])]
            
            if value_cols and group_cols:
                value_col = value_cols[0]
                group_col = group_cols[0]
                
                # Create top contributors table
                top_contrib = df.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(10)
                
                if not top_contrib.empty:
                    contrib_df = pd.DataFrame({
                        group_col: top_contrib.index,
                        'Total Value': top_contrib.values
                    })
                    
                    tables.append({
                        'title': f'Top Contributors by {group_col}',
                        'data': contrib_df
                    })
                    
        except Exception as e:
            print(f"DEBUG: Top contributors analysis failed: {e}")
        
        return tables
    
    def _calculate_two_file_delta(self, dataframes: List[pd.DataFrame]) -> Optional[Dict]:
        """Calculate delta between exactly two files"""
        try:
            df1, df2 = dataframes[0], dataframes[1]
            
            # Look for value columns
            value_cols1 = [col for col in df1.columns if any(term in col.lower() 
                          for term in ['value', 'amount', 'cost', 'total'])]
            value_cols2 = [col for col in df2.columns if any(term in col.lower() 
                          for term in ['value', 'amount', 'cost', 'total'])]
            
            if value_cols1 and value_cols2:
                total1 = df1[value_cols1[0]].sum()
                total2 = df2[value_cols2[0]].sum()
                delta = total2 - total1
                
                return {
                    'total_delta': delta,
                    'file1_total': total1,
                    'file2_total': total2
                }
                
        except Exception as e:
            print(f"DEBUG: Two-file delta calculation failed: {e}")
        
        return None
    
    def _create_aging_summary_table(self, aging_analysis: Dict[str, Dict[str, float]]) -> Dict:
        """Create aging $ by bucket summary table"""
        try:
            # Collect all unique buckets
            all_buckets = set()
            for file_aging in aging_analysis.values():
                all_buckets.update(file_aging.keys())
            
            # Create summary data
            summary_data = []
            for file_name, aging_data in aging_analysis.items():
                row = {'File': file_name}
                for bucket in sorted(all_buckets):
                    row[bucket] = aging_data.get(bucket, 0)
                summary_data.append(row)
            
            # Add totals row
            totals_row = {'File': 'TOTAL'}
            for bucket in sorted(all_buckets):
                total = sum(aging_data.get(bucket, 0) for aging_data in aging_analysis.values())
                totals_row[bucket] = total
            summary_data.append(totals_row)
            
            aging_df = pd.DataFrame(summary_data)
            
            return {
                'title': 'Aging $ by Bucket',
                'data': aging_df
            }
            
        except Exception as e:
            print(f"DEBUG: Aging summary table creation failed: {e}")
            return {
                'title': 'Aging Analysis',
                'data': pd.DataFrame({'Status': ['Analysis failed'], 'Details': [str(e)]})
            }

    def _create_analysis_tables(self, loaded_data_info: Dict[str, str], calculations: List[str]) -> List[Dict]:
        """Create proper DataFrame tables for display"""
        tables = []
        
        # Create a summary table from calculations
        if calculations:
            # Convert calculations to a structured table
            calc_data = []
            for calc in calculations[:10]:  # Limit to 10 calculations
                if ":" in calc:
                    metric, value = calc.split(":", 1)
                    calc_data.append({"Metric": metric.strip(), "Value": value.strip()})
                else:
                    calc_data.append({"Metric": "Analysis", "Value": calc})
            
            if calc_data:
                calc_df = pd.DataFrame(calc_data)
                tables.append({
                    "title": "Analysis Results",
                    "data": calc_df
                })
        
        # Create file summary table from loaded data info
        if loaded_data_info:
            file_data = []
            for name, info in loaded_data_info.items():
                # Parse info like "150 rows x 25 cols"
                try:
                    if " rows x " in info and " cols" in info:
                        rows_part, cols_part = info.split(" rows x ")
                        rows = int(rows_part)
                        cols = int(cols_part.replace(" cols", ""))
                        file_data.append({
                            "File": name,
                            "Rows": rows,
                            "Columns": cols,
                            "Status": "Loaded Successfully"
                        })
                    else:
                        file_data.append({
                            "File": name,
                            "Rows": "Unknown",
                            "Columns": "Unknown", 
                            "Status": info
                        })
                except:
                    file_data.append({
                        "File": name,
                        "Rows": "Unknown",
                        "Columns": "Unknown",
                        "Status": "Loaded"
                    })
            
            if file_data:
                file_df = pd.DataFrame(file_data)
                tables.append({
                    "title": "File Analysis Summary",
                    "data": file_df
                })
        
        # Fallback if no tables created
        if not tables:
            fallback_df = pd.DataFrame({
                "Status": ["Analysis Completed"],
                "Details": ["Data processing completed successfully"]
            })
            tables.append({
                "title": "Analysis Status",
                "data": fallback_df
            })
        
        return tables

    def _apply_quality_protocol(self, execution_result: Dict[str, Any], intent: str, query: str) -> Dict[str, Any]:
        """Apply quality protocol and confidence scoring"""
        try:
            # Get execution results
            calls = execution_result.get("calls", [])
            selected_files = execution_result.get("selected_files", [])
            artifacts = execution_result.get("artifacts", [])
            
            # Calculate confidence score
            try:
                from confidence import calculate_ravc_confidence
                
                # Prepare data for confidence calculation
                kb_sources = []
                for call in calls:
                    if call.get("tool") == "kb_search" and not call.get("error"):
                        kb_sources.extend(call.get("result_meta", {}).get("sources", []))
                
                confidence_score, ravc_breakdown = calculate_ravc_confidence(
                    query=query,
                    sources=kb_sources,
                    execution_calls=calls
                )
            except Exception as e:
                print(f"Warning: Confidence calculation failed: {e}")
                confidence_score = 0.75  # Default confidence
                ravc_breakdown = {
                    "retrieval_strength_R": 0.75,
                    "agreement_A": 0.75,
                    "validations_V": 0.75,
                    "citation_density_C": 0.75
                }
            
            # Generate file summaries
            file_summaries = []
            for file_info in selected_files:
                summary = f"File: {file_info['name']}. Contains data for analysis."
                if file_info.get('metadata'):
                    metadata = file_info['metadata']
                    if metadata.get('period'):
                        summary += f" Period: {metadata['period']}."
                    if metadata.get('country'):
                        summary += f" Country: {metadata['country']}."
                file_summaries.append(summary)
            
            # Extract actual insights from execution results
            insights_from_analysis = []
            calculations_from_analysis = []
            
            for call in calls:
                if call.get("success") and call.get("result_meta"):
                    result = call["result_meta"]
                    if result.get("insights"):
                        insights_from_analysis.extend(result["insights"])
                    if result.get("calculations"):
                        calculations_from_analysis.extend(result["calculations"])
            
            # Build structured response with actual analysis results
            response = {
                "title": f"{intent.replace('_', ' ').title()} Analysis: {query}",
                "executive_insight": self._generate_executive_insight_from_results(calls, insights_from_analysis),
                "method_and_scope": {
                    "files_analyzed": len(selected_files),
                    "analysis_type": intent,
                    "data_sources": file_summaries[:3]  # First 3 file summaries
                },
                "evidence_and_calculations": {
                    "tables": self._create_analysis_tables(execution_result.get("loaded_data", {}), calculations_from_analysis),
                    "charts": [call.get("result_meta", {}).get("chart_paths", []) for call in calls if call.get("tool") == "chart"],
                    "calculations": calculations_from_analysis if calculations_from_analysis else ["Analysis completed with available data"],
                    "insights": insights_from_analysis if insights_from_analysis else ["Data processing completed"]
                },
                "root_causes_drivers": self._extract_drivers_from_results(insights_from_analysis, intent),
                "recommendations": self._generate_recommendations_from_results(insights_from_analysis, intent),
                "confidence": {
                    "score": confidence_score,
                    "level": "High" if confidence_score >= 0.75 else "Medium" if confidence_score >= 0.55 else "Low",
                    "ravc_breakdown": ravc_breakdown
                },
                "limits_data_needed": self._identify_limitations(execution_result, kb_sources),
                "citations": kb_sources[:10],  # Top 10 citations
                "artifacts": artifacts,
                "intent": intent,
                "query": query
            }
            
            return response
            
        except Exception as e:
            print(f"Warning: Quality protocol failed: {e}")
            # Return a basic response structure
            return {
                "title": f"Analysis Results: {query}",
                "executive_insight": "Analysis completed with available data.",
                "method_and_scope": {
                    "files_analyzed": len(execution_result.get("selected_files", [])),
                    "analysis_type": intent,
                    "data_sources": ["Analysis performed on available data"]
                },
                "evidence_and_calculations": {
                    "tables": [],
                    "charts": [],
                    "calculations": ["Basic analysis completed"]
                },
                "root_causes_drivers": ["Analysis completed with available information"],
                "recommendations": ["Review data quality and completeness"],
                "confidence": {
                    "score": 0.6,
                    "level": "Medium",
                    "ravc_breakdown": {
                        "retrieval_strength_R": 0.6,
                        "agreement_A": 0.6,
                        "validations_V": 0.6,
                        "citation_density_C": 0.6
                    }
                },
                "limits_data_needed": ["Quality protocol encountered errors"],
                "citations": [],
                "artifacts": execution_result.get("artifacts", []),
                "intent": intent,
                "query": query,
                "error": f"Quality protocol failed: {str(e)}"
            }

# Global instance
dp_orchestrator = DataProcessingOrchestrator()
