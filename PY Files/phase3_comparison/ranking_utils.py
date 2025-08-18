# ranking_utils.py

import pandas as pd
import os
import json
from datetime import datetime

# Enterprise foundation imports
from constants import DATA_ROOT, COMPARES
from path_utils import join_root, canon_path

# Assistant integration
try:
    from assistant_bridge import run_query as assistants_answer
    ASSISTANT_AVAILABLE = True
except ImportError:
    ASSISTANT_AVAILABLE = False
    def assistants_answer(query, context=""): return "Assistant not available"

def rank_task(df: pd.DataFrame, entity_type: str = None, top_n: int = 10, filters: dict = None, output_folder: str = None) -> dict:
    """
    Enterprise-grade ranking with Assistant-driven analysis and artifact generation.
    Returns paths to generated artifacts, not local DataFrames.

    Args:
        df (pd.DataFrame): Data to rank (can be comparison results or raw data).
        entity_type (str, optional): 'wip', 'inventory', 'finance', or None for auto-detection.
        top_n (int): Number of top results to return.
        filters (dict): Optional filter conditions {column: [values]}.
        output_folder (str, optional): Output directory. Defaults to enterprise COMPARES.

    Returns:
        dict: Paths to ranking artifacts (Excel, JSON, summary).
    """
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be None or empty")
    
    # Enterprise path handling
    if output_folder is None:
        output_folder = COMPARES
    output_folder = canon_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    df_filtered = df.copy()

    # Apply filters if provided
    if filters:
        for col, values in filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col].isin(values)]

    # Assistant-driven entity type detection if not provided
    if entity_type is None:
        columns = list(df_filtered.columns)
        if ASSISTANT_AVAILABLE:
            try:
                context = {
                    "columns": columns[:20],
                    "sample_data": df_filtered.head(3).to_dict("records"),
                    "row_count": len(df_filtered)
                }
                
                entity_response = assistants_answer(
                    f"Based on these columns: {columns}, determine the entity type. "
                    f"Options: 'wip' (jobs/aging), 'inventory' (parts/stock), 'finance' (GL/costs), 'parts' (general parts analysis). "
                    f"Respond with just the entity type.",
                    context=json.dumps(context)[:1500]
                )
                
                entity_type = entity_response.lower().strip()
                if entity_type not in ['wip', 'inventory', 'finance', 'parts']:
                    entity_type = 'parts'  # Default fallback
                    
            except Exception:
                entity_type = 'parts'  # Fallback
        else:
            # Heuristic detection without Assistant
            columns_lower = [col.lower() for col in columns]
            if any(keyword in ' '.join(columns_lower) for keyword in ['job', 'wip', 'aging']):
                entity_type = 'wip'
            elif any(keyword in ' '.join(columns_lower) for keyword in ['gl', 'account', 'financial']):
                entity_type = 'finance'
            elif any(keyword in ' '.join(columns_lower) for keyword in ['part', 'inventory', 'stock']):
                entity_type = 'inventory'
            else:
                entity_type = 'parts'

    # Dynamic column detection for ranking
    numeric_cols = df_filtered.select_dtypes(include=[float, int]).columns.tolist()
    
    # Assistant-driven ranking strategy
    if ASSISTANT_AVAILABLE and numeric_cols:
        try:
            ranking_context = {
                "entity_type": entity_type,
                "numeric_columns": numeric_cols[:10],
                "total_rows": len(df_filtered),
                "column_stats": {col: {"mean": df_filtered[col].mean(), "max": df_filtered[col].max()} 
                               for col in numeric_cols[:5]}
            }
            
            ranking_strategy = assistants_answer(
                f"For {entity_type} ranking with columns {numeric_cols}, which column should be the primary sort? "
                f"Consider business impact and variance. Respond with just the column name.",
                context=json.dumps(ranking_context)[:1500]
            )
            
            # Validate Assistant response
            suggested_col = ranking_strategy.strip()
            if suggested_col in numeric_cols:
                sort_col = suggested_col
            else:
                # Fallback to largest variance column
                sort_col = df_filtered[numeric_cols].var().idxmax() if numeric_cols else numeric_cols[0]
                
        except Exception:
            # Fallback to heuristic column selection
            sort_col = _get_default_sort_column(entity_type, numeric_cols)
    else:
        sort_col = _get_default_sort_column(entity_type, numeric_cols)
    
    if sort_col not in df_filtered.columns:
        if numeric_cols:
            sort_col = numeric_cols[0]  # Use first numeric column
        else:
            raise KeyError(f"No suitable numeric columns found for ranking")

    # Perform ranking
    try:
        ranked = df_filtered.sort_values(by=sort_col, key=abs, ascending=False)
    except Exception:
        ranked = df_filtered.sort_values(by=sort_col, ascending=False)
    
    top_ranked = ranked.head(top_n)

    # Generate artifacts with enterprise paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = f"ranking_{entity_type}_{timestamp}"
    
    # Excel file
    excel_path = join_root(output_folder, f"{file_prefix}.xlsx")
    top_ranked.to_excel(excel_path, index=False)
    
    # JSON file for programmatic access
    json_path = join_root(output_folder, f"{file_prefix}.json")
    ranking_data = {
        "metadata": {
            "entity_type": entity_type,
            "sort_column": sort_col,
            "top_n": top_n,
            "total_rows": len(df_filtered),
            "ranking_timestamp": datetime.now().isoformat(),
            "filters_applied": filters or {}
        },
        "rankings": top_ranked.to_dict("records")
    }
    
    with open(json_path, 'w') as f:
        json.dump(ranking_data, f, indent=2, default=str)
    
    # Assistant-enhanced summary
    summary_text = _generate_ranking_summary(top_ranked, entity_type, sort_col, top_n)
    
    summary_path = join_root(output_folder, f"{file_prefix}_summary.md")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    return {
        "excel_file": excel_path,
        "json_file": json_path,
        "summary_file": summary_path,
        "entity_type": entity_type,
        "sort_column": sort_col,
        "records_ranked": len(top_ranked)
    }

def _get_default_sort_column(entity_type: str, numeric_cols: list) -> str:
    """Get default sort column based on entity type and available columns."""
    # Entity-specific preferred columns
    preferred_cols = {
        'wip': ['delta', 'value_delta', 'aging_value', 'total_value', 'amount'],
        'inventory': ['value_delta', 'quantity_delta', 'extended_cost', 'value', 'cost'],
        'finance': ['value_delta', 'amount', 'balance', 'total', 'value'],
        'parts': ['extended_cost', 'value', 'cost', 'amount', 'total']
    }
    
    entity_prefs = preferred_cols.get(entity_type, preferred_cols['parts'])
    
    # Find first preferred column that exists
    for pref_col in entity_prefs:
        if pref_col in numeric_cols:
            return pref_col
    
    # Fallback to first numeric column
    return numeric_cols[0] if numeric_cols else None

def _generate_ranking_summary(ranked_df: pd.DataFrame, entity_type: str, sort_col: str, top_n: int) -> str:
    """Generate Assistant-enhanced ranking summary."""
    summary_lines = [
        f"# {entity_type.title()} Ranking Analysis",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Ranked by:** {sort_col}",
        f"**Top:** {top_n} records",
        "",
        "## Key Findings"
    ]
    
    if ASSISTANT_AVAILABLE and not ranked_df.empty:
        try:
            # Prepare context for Assistant analysis
            top_3 = ranked_df.head(3)
            context = {
                "entity_type": entity_type,
                "sort_column": sort_col,
                "top_records": top_3.to_dict("records"),
                "value_range": {
                    "max": ranked_df[sort_col].max(),
                    "min": ranked_df[sort_col].min(),
                    "mean": ranked_df[sort_col].mean()
                }
            }
            
            insights = assistants_answer(
                f"Analyze this {entity_type} ranking data and provide 3 key business insights about the top performers. "
                f"Focus on actionable observations for supply chain management.",
                context=json.dumps(context, default=str)[:2000]
            )
            
            summary_lines.append(insights)
            
        except Exception:
            # Fallback to basic statistical summary
            if len(ranked_df) > 0:
                summary_lines.extend([
                    f"- **Top performer:** {ranked_df.iloc[0].get('part_no', ranked_df.iloc[0].name)} with {sort_col}: {ranked_df[sort_col].iloc[0]:,.2f}",
                    f"- **Value range:** {ranked_df[sort_col].min():,.2f} to {ranked_df[sort_col].max():,.2f}",
                    f"- **Average {sort_col}:** {ranked_df[sort_col].mean():,.2f}"
                ])
            else:
                summary_lines.append("- **No data available for analysis**")
    else:
        # Basic summary without Assistant
        if not ranked_df.empty and len(ranked_df) > 0:
            summary_lines.extend([
                f"- **Records analyzed:** {len(ranked_df)}",
                f"- **Top value:** {ranked_df[sort_col].iloc[0]:,.2f}",
                f"- **Range:** {ranked_df[sort_col].min():,.2f} to {ranked_df[sort_col].max():,.2f}"
            ])
        else:
            summary_lines.append("- **No data available for analysis**")
    
    summary_lines.extend([
        "",
        "## Data Quality",
        f"- **Total records:** {len(ranked_df)}",
        f"- **Columns analyzed:** {len(ranked_df.columns)}",
        f"- **Null values:** {ranked_df.isnull().sum().sum()}",
        "",
        "*Generated by SCIE Ethos Enterprise Ranking System*"
    ])
    
    return "\n".join(summary_lines)
