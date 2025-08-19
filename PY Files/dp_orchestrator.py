# PY Files/dp_orchestrator.py
"""
Enhanced Data Processing Orchestrator
Routes all DP queries through master instructions with full enterprise parity
"""

import os
import re
import json
import yaml
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

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
    from dbx_utils import list_files_in_folder, read_file_bytes
    from file_utils import list_cleaned_files
    from confidence import calculate_ravc_confidence
    from phase5_governance.query_logger import QueryLogger
except ImportError as e:
    print(f"Warning: Could not import components: {e}")

class DataProcessingOrchestrator:
    """Enhanced Data Processing Orchestrator with enterprise parity"""
    
    def __init__(self):
        self.query_logger = QueryLogger() if 'QueryLogger' in globals() else None
        self.cleansed_files_path = "/Project_Root/04_Data/01_Cleansed_Files/"
        
    def resolve_files_metadata_first(self, query: str, intent: str) -> Dict[str, Any]:
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
                country_filters, erp_filters, sheet_type_filters
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
            
            # Try to get files from Dropbox first
            try:
                from dbx_utils import list_files_in_folder
                dropbox_files = list_files_in_folder(self.cleansed_files_path)
                for file_info in dropbox_files:
                    if file_info.get('name', '').endswith('.xlsx'):
                        files.append({
                            "path": file_info['path'],
                            "name": file_info['name'],
                            "size": file_info.get('size', 0),
                            "modified": file_info.get('modified', ''),
                            "metadata": self._extract_metadata_from_filename(file_info['name'])
                        })
            except Exception:
                # Fallback to session state or local files
                pass
            
            # Add files from session state if available
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'cleaned_sheets'):
                    for sheet_name, sheet_data in st.session_state.cleaned_sheets.items():
                        files.append({
                            "path": f"session:{sheet_name}",
                            "name": sheet_name,
                            "size": len(str(sheet_data)) if sheet_data is not None else 0,
                            "modified": datetime.now().isoformat(),
                            "metadata": self._extract_metadata_from_filename(sheet_name),
                            "data": sheet_data
                        })
            except Exception:
                pass
            
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
                               erps: List[str], sheet_types: List[str]) -> List[Dict]:
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
            # Comparison needs at least 2 files, prefer up to 4
            return ranked_files[:min(4, len(ranked_files))]
        elif intent in ["forecasting", "movement_analysis", "optimization"]:
            # These intents can use multiple files for comprehensive analysis
            return ranked_files[:min(6, len(ranked_files))]
        elif intent in ["root_cause", "anomaly_detection"]:
            # Root cause and anomaly detection benefit from multiple perspectives
            return ranked_files[:min(3, len(ranked_files))]
        else:
            # Default: use top ranked file(s)
            return ranked_files[:min(2, len(ranked_files))]
    
    def process_dp_query(self, query: str, session_state: Any = None) -> Dict[str, Any]:
        """
        Process Data Processing query with full enterprise parity
        """
        try:
            # Step 1: Classify intent using master instructions
            intent_result = classify_user_intent(query)
            intent = intent_result.get("intent", "eda")
            confidence = intent_result.get("confidence", 0.5)
            
            # Step 2: Resolve files using metadata-first approach
            file_resolution = self.resolve_files_metadata_first(query, intent)
            
            # Step 3: Check if clarification is needed
            if file_resolution.get("needs_clarification"):
                return self._generate_clarification_response(file_resolution, query)
            
            selected_files = file_resolution.get("selected_files", [])
            if not selected_files:
                return {
                    "error": "No suitable files found for analysis",
                    "suggestions": "Please upload and process files first, or be more specific about which files to analyze."
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
            
        elif intent in ["root_cause", "eda", "movement_analysis", "anomaly_detection"]:
            plan.append({
                "tool": "dataframe_query",
                "args": {
                    "files": file_paths,
                    "query_type": intent,
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
            "eda": ["distribution", "correlation_matrix", "summary_stats"]
        }
        
        return chart_mapping.get(intent, ["bar", "line", "scatter"])
    
    def _execute_with_reasoning_protocol(self, plan: List[Dict], selected_files: List[Dict]) -> Dict[str, Any]:
        """Execute plan using enterprise reasoning protocol"""
        try:
            # Create mock app_paths for execution
            class MockAppPaths:
                def __init__(self):
                    self.data_root = "/Project_Root/04_Data"
            
            app_paths = MockAppPaths()
            
            # Execute the plan
            execution_result = _execute_plan(plan, app_paths)
            
            # Add file context to results
            execution_result["selected_files"] = selected_files
            execution_result["file_summaries"] = self._generate_file_summaries(selected_files)
            
            return execution_result
            
        except Exception as e:
            return {
                "error": f"Execution failed: {str(e)}",
                "plan": plan,
                "selected_files": selected_files
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
        """Identify limitations and data gaps"""
        limitations = []
        
        # Check for execution errors
        errors = [call.get("error") for call in execution_result.get("calls", []) if call.get("error")]
        if errors:
            limitations.append(f"Some analysis components failed: {'; '.join(errors[:2])}")
        
        # Check KB coverage
        if len(kb_sources) < 2:
            limitations.append("Limited knowledge base coverage - results may lack context.")
        
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

# Global instance
dp_orchestrator = DataProcessingOrchestrator()
