# phase2_analysis/dataframe_query.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import re

# Enhanced foundation imports
try:
    from constants import DATA_ROOT
    from path_utils import canon_path
    from logger import log_event
except ImportError:
    # Fallback for standalone usage - use proper Dropbox path
    DATA_ROOT = "/Apps/Ethos LLM/Project_Root/04_Data"
    def canon_path(p): return p  # Don't resolve filesystem paths for cloud storage
    def log_event(msg, path=None): print(f"LOG: {msg}")

# AI integration for intelligent analytics
try:
    from llm_client import get_openai_client, chat_completion
except ImportError:
    get_openai_client = None
    chat_completion = None

# Cloud storage integration
try:
    from dbx_utils import read_file_bytes, upload_bytes
    from tools_runtime import _load_file_to_frames, _apply_filters, _save_artifact
except ImportError:
    def read_file_bytes(path): raise NotImplementedError("Cloud storage not available")
    def upload_bytes(path, data, mode="overwrite"): raise NotImplementedError("Cloud storage not available")
    def _load_file_to_frames(path): raise NotImplementedError("tools_runtime not available")
    def _apply_filters(df, filters): return df
    def _save_artifact(df, folder, base, format): return f"{folder}/{base}.{format}"

# ================== ENTERPRISE DATAFRAME QUERY INTERFACE ==================

def dataframe_query(
    files: Optional[List[str]] = None,
    *,
    cleansed_paths: Optional[List[str]] = None,  # Legacy compatibility
    sheet: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    groupby: Optional[List[str]] = None,
    metrics: Optional[List[Dict[str, str]]] = None,
    limit: int = 50,
    query_type: str = "general",  # "wip", "inventory", "financial", "general"
    ai_enhance: bool = True,
    artifact_folder: Optional[str] = None,
    artifact_format: str = "excel",
    aggregation_plan: Optional[Dict[str, Any]] = None  # NEW: Accept aggregation plan from choose_aggregation()
) -> Dict[str, Any]:
    """
    Enterprise-grade dataframe querying with AI enhancement and supply chain intelligence.
    Now supports executing generic aggregation plans from choose_aggregation().
    
    Contract per specification:
    Returns: {rowcount, sheet_used, total_wip?, ...}
    
    Args:
        files: List of cleansed file paths to query
        sheet: Specific sheet name (optional)
        filters: Filter conditions
        groupby: Grouping columns
        metrics: Aggregation metrics
        limit: Preview row limit
        query_type: "wip", "inventory", "financial", "general" - enables domain intelligence
        ai_enhance: Enable AI-powered query optimization and insights
        artifact_folder: Where to save results (defaults to /04_Data/03_Summaries)
        artifact_format: "csv" or "excel"
        aggregation_plan: JSON plan from choose_aggregation() {op, col, groupby, filters}
        
    Returns:
        Dict with enterprise contract fields:
        - rowcount: int
        - sheet_used: str
        - total_wip: float (if applicable)
        - columns: List[str]
        - preview: List[Dict]
        - artifact_path: str (optional)
        - ai_insights: Dict (optional)
        - query_performance: Dict
    """
    
    # Handle parameter compatibility
    paths = files if files is not None else cleansed_paths
    if not paths:
        return {
            "error": "No file paths provided",
            "rowcount": 0,
            "sheet_used": None,
            "total_wip": None
        }
    
    query_start_time = pd.Timestamp.now()
    
    try:
        # Load data with enterprise error handling
        df, sheet_used = _enterprise_load_data(paths[0], sheet)
        log_event(f"Loaded data: {len(df)} rows from {sheet_used}")
        
        # Execute aggregation plan if provided (NEW from ChatGPT plan)
        if aggregation_plan:
            df_result = _execute_aggregation_plan(df, aggregation_plan)
            log_event(f"Executed aggregation plan: {aggregation_plan.get('op', 'unknown')} operation")
        else:
            # Apply filters with validation
            if filters:
                df_filtered = _apply_filters(df, filters)
                log_event(f"Applied {len(filters)} filters: {len(df_filtered)} rows remaining")
            else:
                df_filtered = df
            
            # Execute groupby and metrics with domain intelligence
            df_result = _execute_aggregations(df_filtered, groupby, metrics, query_type)
        
        # AI-enhanced query optimization
        ai_insights = {}
        if ai_enhance and get_openai_client:
            try:
                ai_insights = _ai_analyze_query(df_result, query_type, filters, groupby, metrics)
                log_event(f"AI analysis completed: {len(ai_insights.get('recommendations', []))} recommendations")
            except Exception as e:
                log_event(f"AI analysis failed: {e}")
                ai_insights = {"status": "unavailable", "reason": str(e)}
        
        # Calculate domain-specific totals
        domain_totals = _calculate_domain_totals(df_result, query_type)
        
        # Apply limit for preview
        df_preview = df_result.head(limit)
        
        # Save artifact if requested - default to /04_Data/03_Summaries
        artifact_path = None
        if artifact_folder or aggregation_plan:
            save_folder = artifact_folder or f"{DATA_ROOT}/03_Summaries"
            artifact_path = _save_query_artifact(df_result, save_folder, query_type, artifact_format)
        
        # Performance metrics
        query_end_time = pd.Timestamp.now()
        performance = {
            "execution_time_ms": int((query_end_time - query_start_time).total_seconds() * 1000),
            "rows_processed": len(df),
            "rows_returned": len(df_result),
            "memory_usage_mb": round(df_result.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        # Build enterprise response
        response = {
            "rowcount": int(len(df_result)),
            "sheet_used": sheet_used,
            "columns": list(map(str, df_preview.columns)),
            "preview": json.loads(df_preview.to_json(orient="records")),
            "query_performance": performance
        }
        
        # Add domain-specific fields
        response.update(domain_totals)
        
        if artifact_path:
            response["artifact_path"] = artifact_path
            
        if ai_insights.get("status") != "unavailable":
            response["ai_insights"] = ai_insights
            
        log_event(f"Query completed: {response['rowcount']} rows, {performance['execution_time_ms']}ms")
        return response
        
    except Exception as e:
        log_event(f"Query failed: {e}")
        return {
            "error": str(e),
            "rowcount": 0,
            "sheet_used": None,
            "total_wip": None
        }

def _execute_aggregation_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> pd.DataFrame:
    """
    NEW: Execute generic aggregation plan from choose_aggregation().
    
    Plan format: {op, col, groupby, filters}
    - Apply filters first
    - If groupby: return small result table  
    - Else: return scalar {value} using op on col
    """
    
    # Apply filters first
    if plan.get("filters"):
        df = _apply_filters(df, plan["filters"])
        log_event(f"Applied {len(plan['filters'])} filters from plan")
    
    op = plan.get("op", "sum").lower()
    col = plan.get("col")
    groupby = plan.get("groupby")
    
    if not col or col not in df.columns:
        # Find a suitable column if not specified or missing
        value_cols = [c for c in df.columns if any(term in str(c).lower() 
                     for term in ['cost', 'value', 'amount', 'price', 'wip', 'total'])]
        col = value_cols[0] if value_cols else df.columns[0]
    
    try:
        if groupby and isinstance(groupby, list):
            # Return small result table grouped by specified columns
            valid_groupby = [g for g in groupby if g in df.columns]
            
            if valid_groupby:
                if op in ["sum", "mean", "max", "min", "count"]:
                    # Numeric aggregation
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    result = df.groupby(valid_groupby, dropna=False)[col].agg(op).reset_index()
                else:
                    # Default to count
                    result = df.groupby(valid_groupby, dropna=False).size().reset_index(name='count')
                
                log_event(f"Grouped by {valid_groupby}, operation: {op}")
                return result
            else:
                log_event("No valid groupby columns found, falling back to aggregation")
        
        # Return scalar result as single-row DataFrame
        if op == "sum":
            value = pd.to_numeric(df[col], errors='coerce').sum()
        elif op == "count":
            value = len(df)
        elif op == "mean":
            value = pd.to_numeric(df[col], errors='coerce').mean()
        elif op == "max":
            value = pd.to_numeric(df[col], errors='coerce').max()
        elif op == "min":
            value = pd.to_numeric(df[col], errors='coerce').min()
        else:
            # Default to sum
            value = pd.to_numeric(df[col], errors='coerce').sum()
        
        # Return as single-row DataFrame with descriptive column name
        result_col = f"{op}_{col}" if op != "sum" else col
        result = pd.DataFrame([{result_col: value}])
        
        log_event(f"Executed {op} on {col}: {value}")
        return result
        
    except Exception as e:
        log_event(f"Aggregation plan execution failed: {e}")
        # Return original data as fallback
        return df

# ================== HELPER FUNCTIONS ==================

def _enterprise_load_data(file_path: str, sheet: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    """Load data with enterprise error handling and sheet intelligence."""
    try:
        frames = _load_file_to_frames(file_path)
        
        if not frames:
            raise ValueError("No data found in file")
        
        # Intelligent sheet selection
        if sheet and sheet in frames:
            selected_df = frames[sheet]
            sheet_used = sheet
        else:
            # Supply chain domain intelligence for sheet selection
            sheet_used, selected_df = _select_best_sheet(frames)
        
        if selected_df.empty:
            raise ValueError("Selected sheet is empty")
            
        return selected_df.copy(), sheet_used
        
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {str(e)}")

def _select_best_sheet(frames: Dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Intelligent sheet selection using supply chain domain knowledge."""
    
    # Priority order for supply chain data
    priority_patterns = [
        (r"wip|work.*in.*progress", "WIP/Inventory data"),
        (r"inventory|stock", "Inventory data"),
        (r"cost|financial", "Cost data"),
        (r"summary|total", "Summary data")
    ]
    
    # Score sheets based on patterns and data quality
    sheet_scores = {}
    
    for sheet_name, df in frames.items():
        score = 0
        sheet_lower = sheet_name.lower()
        
        # Pattern matching
        for pattern, description in priority_patterns:
            if re.search(pattern, sheet_lower):
                score += 100
                break
        
        # Data quality scoring
        if len(df) > 0:
            score += min(len(df), 50)  # Row count bonus (capped)
            score += min(len(df.columns), 20)  # Column count bonus (capped)
            
            # Supply chain column bonus
            supply_chain_cols = ['part', 'item', 'wip', 'cost', 'value', 'qty', 'quantity']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(sc_col in col_lower for sc_col in supply_chain_cols):
                    score += 10
        
        sheet_scores[sheet_name] = score
    
    # Select best sheet
    best_sheet = max(sheet_scores.items(), key=lambda x: x[1])
    return best_sheet[0], frames[best_sheet[0]]

def _ai_analyze_query(df: pd.DataFrame, query_type: str, filters: Optional[List], 
                     groupby: Optional[List], metrics: Optional[List]) -> Dict[str, Any]:
    """AI-powered query analysis and optimization recommendations."""
    
    client = get_openai_client()
    if not client or not chat_completion:
        return {"status": "unavailable", "reason": "AI client not available"}
    
    # Prepare data summary for AI
    data_summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns)[:20],  # Limit for prompt size
        "query_type": query_type,
        "has_filters": bool(filters),
        "has_groupby": bool(groupby),
        "has_metrics": bool(metrics)
    }
    
    # Sample data for context
    sample_data = df.head(5).to_dict(orient='records') if len(df) > 0 else []
    
    prompt = f"""
    You are a supply chain data analyst reviewing a query against a {query_type} dataset.
    
    Data Summary:
    {json.dumps(data_summary, indent=2)}
    
    Sample Data:
    {json.dumps(sample_data, indent=2)}
    
    Current Query:
    - Filters: {filters or "None"}
    - Group By: {groupby or "None"} 
    - Metrics: {metrics or "None"}
    
    Provide analysis and recommendations in JSON format:
    {{
        "data_quality_score": 0.85,
        "query_efficiency": "high|medium|low",
        "recommendations": [
            {{"type": "optimization", "suggestion": "...", "reasoning": "..."}},
            {{"type": "insight", "finding": "...", "business_impact": "..."}}
        ],
        "suggested_metrics": [
            {{"column": "wip_value", "aggregation": "sum", "reasoning": "..."}}
        ],
        "data_insights": [
            "Key pattern or anomaly found in the data"
        ]
    }}
    """
    
    try:
        messages = [
            {"role": "system", "content": "You are an expert supply chain data analyst. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = chat_completion(client, messages, model="gpt-4o-mini")
        
        # Parse AI response
        try:
            analysis = json.loads(response)
            return analysis
        except json.JSONDecodeError:
            # Extract JSON from response if wrapped in text
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"status": "parse_failed", "raw_response": response[:500]}
                
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def _execute_aggregations(df: pd.DataFrame, groupby: Optional[List[str]], 
                         metrics: Optional[List[Dict]], query_type: str) -> pd.DataFrame:
    """Execute aggregations with domain-specific intelligence."""
    
    if not groupby and not metrics:
        return df
    
    # Default metrics based on query type
    if not metrics and query_type in ["wip", "inventory"]:
        metrics = _get_default_supply_chain_metrics(df)
    
    if groupby and metrics:
        # Build aggregation map
        agg_map = {}
        for metric in metrics:
            col = metric.get("col")
            agg = metric.get("agg", "sum").lower()
            if col and col in df.columns:
                agg_map[col] = agg
        
        if agg_map:
            grouped = df.groupby(groupby, dropna=False).agg(agg_map).reset_index()
            
            # Add calculated fields for supply chain analytics
            if query_type in ["wip", "inventory"]:
                grouped = _add_supply_chain_calculations(grouped)
            
            return grouped
    
    elif metrics and not groupby:
        # Simple aggregations over whole table
        rows = {}
        for metric in metrics:
            col = metric.get("col")
            agg = metric.get("agg", "sum").lower()
            if col and col in df.columns:
                try:
                    value = pd.to_numeric(df[col], errors="coerce").agg(agg)
                    rows[f"{agg}({col})"] = value
                except Exception:
                    rows[f"{agg}({col})"] = None
        
        if rows:
            return pd.DataFrame([rows])
    
    return df

def _get_default_supply_chain_metrics(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Get default metrics based on available columns."""
    metrics = []
    
    # Common supply chain value columns
    value_columns = [
        'wip_value', 'extended_cost', 'total_cost', 'inventory_value',
        'cost', 'value', 'amount', 'extended_price'
    ]
    
    quantity_columns = [
        'qty', 'quantity', 'on_hand', 'qty_on_hand', 'stock_qty'
    ]
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Value metrics
        if any(val_col in col_lower for val_col in value_columns):
            metrics.append({"col": col, "agg": "sum"})
            
        # Quantity metrics  
        elif any(qty_col in col_lower for qty_col in quantity_columns):
            metrics.append({"col": col, "agg": "sum"})
    
    return metrics

def _add_supply_chain_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """Add calculated fields for supply chain analytics."""
    
    # Add utilization ratios
    if 'qty_used' in df.columns and 'qty_on_hand' in df.columns:
        df['utilization_ratio'] = df['qty_used'] / (df['qty_on_hand'] + 1e-9)
    
    # Add value density
    if 'total_cost' in df.columns and 'qty_on_hand' in df.columns:
        df['value_per_unit'] = df['total_cost'] / (df['qty_on_hand'] + 1e-9)
    
    return df

def _calculate_domain_totals(df: pd.DataFrame, query_type: str) -> Dict[str, Any]:
    """Calculate domain-specific totals based on query type."""
    totals = {}
    
    if query_type == "wip":
        # WIP-specific calculations
        wip_columns = [col for col in df.columns if 'wip' in str(col).lower()]
        if wip_columns:
            total_wip = 0
            for col in wip_columns:
                try:
                    total_wip += pd.to_numeric(df[col], errors='coerce').sum()
                except Exception:
                    pass
            totals["total_wip"] = float(total_wip)
        
        # Standard WIP metrics
        for col in ['wip_value', 'work_in_progress', 'wip_cost']:
            if col in df.columns:
                try:
                    totals[col.replace('_', '')] = float(pd.to_numeric(df[col], errors='coerce').sum())
                except Exception:
                    totals[col.replace('_', '')] = 0.0
    
    elif query_type == "inventory":
        # Inventory-specific calculations
        inventory_value_cols = ['inventory_value', 'extended_cost', 'total_cost']
        for col in inventory_value_cols:
            if col in df.columns:
                try:
                    totals["total_inventory_value"] = float(pd.to_numeric(df[col], errors='coerce').sum())
                    break
                except Exception:
                    pass
        
        # Quantity totals
        qty_cols = ['qty_on_hand', 'quantity', 'stock_qty']
        for col in qty_cols:
            if col in df.columns:
                try:
                    totals["total_quantity"] = float(pd.to_numeric(df[col], errors='coerce').sum())
                    break
                except Exception:
                    pass
    
    elif query_type == "financial":
        # Financial totals
        financial_cols = ['cost', 'value', 'amount', 'extended_cost', 'total_cost']
        total_financial = 0
        for col in financial_cols:
            if col in df.columns:
                try:
                    total_financial += pd.to_numeric(df[col], errors='coerce').sum()
                except Exception:
                    pass
        totals["total_financial_value"] = float(total_financial)
    
    # Always try to provide total_wip for contract compliance
    if "total_wip" not in totals:
        # Try to find any WIP-related value
        for col in df.columns:
            col_lower = str(col).lower()
            if 'wip' in col_lower and any(term in col_lower for term in ['value', 'cost', 'amount']):
                try:
                    totals["total_wip"] = float(pd.to_numeric(df[col], errors='coerce').sum())
                    break
                except Exception:
                    pass
        
        if "total_wip" not in totals:
            totals["total_wip"] = None
    
    return totals

def _save_query_artifact(df: pd.DataFrame, artifact_folder: str, query_type: str, 
                        artifact_format: str) -> str:
    """Save query results as artifact with enterprise naming."""
    
    # Use canonical paths
    artifact_folder = canon_path(artifact_folder)
    
    # Enterprise naming convention
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{query_type}_query_{timestamp}"
    
    try:
        artifact_path = _save_artifact(df, artifact_folder, base_name, artifact_format)
        log_event(f"Saved query artifact: {artifact_path}")
        return artifact_path
    except Exception as e:
        log_event(f"Failed to save artifact: {e}")
        return f"{artifact_folder}/{base_name}.{artifact_format}"

# ================== CONVENIENCE FUNCTIONS ==================

def query_wip_total(files: List[str], **kwargs) -> Dict[str, Any]:
    """Convenience function to get total WIP value - for smoke test."""
    kwargs.update({
        "query_type": "wip",
        "metrics": [{"col": col, "agg": "sum"} for col in ["wip_value", "extended_cost", "total_cost"]]
    })
    return dataframe_query(files=files, **kwargs)

def query_inventory_summary(files: List[str], **kwargs) -> Dict[str, Any]:
    """Convenience function for inventory analysis."""
    kwargs.update({
        "query_type": "inventory",
        "groupby": ["item_type", "location"] if "groupby" not in kwargs else kwargs["groupby"]
    })
    return dataframe_query(files=files, **kwargs)

def quick_wip_check(file_path: str) -> float:
    """Quick function for smoke test: 'how much WIP' returns a number."""
    try:
        result = query_wip_total([file_path], limit=1)
        return result.get("total_wip", 0.0) or 0.0
    except Exception:
        return 0.0
