# comparison_utils.py

import os
import pandas as pd
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

def compare_wip_aging(wip_dfs, output_folder=None):
    """
    Enterprise-grade WIP aging comparison with Assistant-driven insights.
    Returns: dict with artifact paths only (Excel, summary, JSON).
    """
    # Enterprise path handling
    if output_folder is None:
        output_folder = COMPARES
    output_folder = canon_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Dynamic aging column detection
    potential_aging_cols = [col for col in wip_dfs[0].columns if 'aging' in col.lower() or any(day in col for day in ['30', '60', '90', '120', '150', '180'])]
    
    # Fallback to standard naming if available
    standard_aging_columns = [
        '0_30_days', 'aging_31_60_days', 'aging_61_90_days',
        'aging_91_120_days', 'aging_121_150_days',
        'aging_151_180_days', 'over_180_days'
    ]
    
    aging_columns = potential_aging_cols if potential_aging_cols else [col for col in standard_aging_columns if col in wip_dfs[0].columns]
    
    if not aging_columns:
        raise ValueError("No aging columns detected. Expected columns with 'aging' or day ranges (30, 60, 90, etc.)")

    melted_wip = pd.DataFrame()
    for i, df in enumerate(wip_dfs):
        # Ensure required columns exist or create them
        if 'period' not in df.columns:
            df = df.copy()
            df['period'] = f"Period_{i+1}"
        if 'source_file' not in df.columns:
            df['source_file'] = f"file_{i+1}"
        
        # Dynamic ID columns detection
        id_columns = []
        for potential_id in ['job_no', 'job_name', 'part_no', 'item_id']:
            if potential_id in df.columns:
                id_columns.append(potential_id)
        
        if not id_columns:
            # Create a generic ID if none found
            df['record_id'] = df.index
            id_columns = ['record_id']
        
        melted = df.melt(
            id_vars=id_columns + ['period', 'source_file'],
            value_vars=aging_columns,
            var_name='aging_bucket',
            value_name='value'
        )
        melted_wip = pd.concat([melted_wip, melted], ignore_index=True)

    periods = sorted(melted_wip['period'].unique())
    if len(periods) < 2:
        raise ValueError("At least two periods required for comparison.")
    p1, p2 = periods[0], periods[-1]

    # Dynamic pivot based on available ID columns  
    index_cols = [col for col in melted_wip.columns if col in id_columns] + ['aging_bucket']
    
    pivot = melted_wip.pivot_table(
        index=index_cols,
        columns='period',
        values='value',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    pivot['delta'] = pivot[p2] - pivot[p1]
    pivot['delta_pct'] = ((pivot[p2] - pivot[p1]) / (pivot[p1] + 1e-9)) * 100  # Avoid division by zero
    pivot_sorted = pivot.sort_values(by='delta', key=abs, ascending=False)

    # Generate enterprise artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = f"wip_comparison_{p1.lower()}_to_{p2.lower()}_{timestamp}"
    
    # Excel file with enterprise path
    excel_path = join_root(output_folder, f"{file_prefix}.xlsx")
    pivot_sorted.to_excel(excel_path, index=False)
    
    # JSON file for programmatic access
    json_path = join_root(output_folder, f"{file_prefix}.json")
    comparison_data = {
        "metadata": {
            "comparison_type": "wip_aging",
            "periods": periods,
            "aging_columns": aging_columns,
            "total_delta": float(pivot['delta'].sum()),
            "records_compared": len(pivot_sorted),
            "timestamp": datetime.now().isoformat()
        },
        "data": pivot_sorted.to_dict("records")
    }
    
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)

    # Assistant-enhanced summary
    summary_md = _generate_wip_summary(pivot_sorted, p1, p2, aging_columns)
    md_path = join_root(output_folder, f"{file_prefix}_summary.md")
    with open(md_path, "w") as f:
        f.write(summary_md)

    return {
        "excel_file": excel_path,
        "json_file": json_path, 
        "summary_file": md_path,
        "comparison_type": "wip_aging",
        "periods_compared": [p1, p2],
        "total_delta": float(pivot['delta'].sum())
    }

def compare_inventory(inventory_dfs, output_folder=None):
    """
    Enterprise-grade inventory comparison with dynamic column detection and Assistant insights.
    """
    # Enterprise path handling
    if output_folder is None:
        output_folder = COMPARES
    output_folder = canon_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    df_all = pd.concat(inventory_dfs, ignore_index=True)
    periods = sorted(df_all['period'].unique())
    if len(periods) < 2:
        raise ValueError("Need at least two periods.")
    p1, p2 = periods[0], periods[-1]

    # Dynamic column detection for inventory data
    id_columns = []
    for potential_id in ['part_number', 'part_no', 'item_id', 'sku']:
        if potential_id in df_all.columns:
            id_columns.append(potential_id)
    
    if not id_columns:
        df_all['item_id'] = df_all.index
        id_columns = ['item_id']
    
    # Dynamic value columns
    value_columns = []
    for potential_val in ['remaining_quantity', 'quantity', 'qty', 'value', 'extended_cost', 'cost']:
        if potential_val in df_all.columns:
            value_columns.append(potential_val)
    
    if not value_columns:
        raise ValueError("No quantity or value columns found for comparison")

    # Include description if available
    groupby_cols = id_columns.copy()
    if 'description' in df_all.columns:
        groupby_cols.append('description')

    df_summary = df_all.groupby(groupby_cols + ['period'])[value_columns].sum().reset_index()
    pivot = df_summary.pivot_table(
        index=groupby_cols,
        columns='period',
        values=value_columns,
        fill_value=0
    )
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
    pivot = pivot.reset_index()

    # Calculate deltas for all value columns
    for col in value_columns:
        if f'{col}_{p1}' in pivot.columns and f'{col}_{p2}' in pivot.columns:
            pivot[f'{col}_delta'] = pivot[f'{col}_{p2}'] - pivot[f'{col}_{p1}']

    # Generate enterprise artifacts  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = f"inventory_comparison_{p1.lower()}_to_{p2.lower()}_{timestamp}"
    
    excel_path = join_root(output_folder, f"{file_prefix}.xlsx")
    pivot.to_excel(excel_path, index=False)
    
    # JSON file
    json_path = join_root(output_folder, f"{file_prefix}.json")
    comparison_data = {
        "metadata": {
            "comparison_type": "inventory",
            "periods": [p1, p2],
            "value_columns": value_columns,
            "records_compared": len(pivot),
            "timestamp": datetime.now().isoformat()
        },
        "data": pivot.to_dict("records")
    }
    
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)

    # Enhanced summary
    summary_md = _generate_inventory_summary(pivot, p1, p2)
    md_path = join_root(output_folder, f"{file_prefix}_summary.md")
    with open(md_path, "w") as f:
        f.write(summary_md)

    return {
        "excel_file": excel_path,
        "json_file": json_path,
        "summary_file": md_path,
        "comparison_type": "inventory",
        "periods_compared": [p1, p2]
    }

def compare_financials(financial_dfs, output_folder=None):
    """
    Enterprise-grade financial comparison with dynamic GL account detection and Assistant insights.
    """
    # Enterprise path handling
    if output_folder is None:
        output_folder = COMPARES
    output_folder = canon_path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    df_all = pd.concat(financial_dfs, ignore_index=True)
    periods = sorted(df_all['period'].unique())
    if len(periods) < 2:
        raise ValueError("Need at least two periods.")
    p1, p2 = periods[0], periods[-1]

    # Dynamic account column detection
    account_col = None
    for potential_col in ['gl_account', 'account', 'account_number', 'gl_code']:
        if potential_col in df_all.columns:
            account_col = potential_col
            break
    
    if account_col is None:
        raise ValueError("No GL account column found")

    # Dynamic value column detection
    value_col = None
    for potential_val in ['value', 'amount', 'balance', 'total']:
        if potential_val in df_all.columns:
            value_col = potential_val
            break
    
    if value_col is None:
        raise ValueError("No value column found")

    summary = df_all.groupby([account_col, 'period'])[[value_col]].sum().reset_index()
    pivot = summary.pivot_table(
        index=account_col,
        columns='period',
        values=value_col,
        fill_value=0
    ).reset_index()

    pivot['value_delta'] = pivot[p2] - pivot[p1]

    # Generate enterprise artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = f"financial_comparison_{p1.lower()}_to_{p2.lower()}_{timestamp}"
    
    excel_path = join_root(output_folder, f"{file_prefix}.xlsx")
    pivot.to_excel(excel_path, index=False)
    
    # JSON file
    json_path = join_root(output_folder, f"{file_prefix}.json")
    comparison_data = {
        "metadata": {
            "comparison_type": "financial",
            "periods": [p1, p2],
            "account_column": account_col,
            "value_column": value_col,
            "total_delta": float(pivot['value_delta'].sum()),
            "accounts_compared": len(pivot),
            "timestamp": datetime.now().isoformat()
        },
        "data": pivot.to_dict("records")
    }
    
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)

    # Enhanced summary
    summary_md = _generate_financial_summary(pivot, p1, p2)
    md_path = join_root(output_folder, f"{file_prefix}_summary.md")
    with open(md_path, "w") as f:
        f.write(summary_md)

    return {
        "excel_file": excel_path,
        "json_file": json_path,
        "summary_file": md_path,
        "comparison_type": "financial",
        "periods_compared": [p1, p2],
        "total_delta": float(pivot['value_delta'].sum())
    }


def _generate_inventory_summary(pivot_df: pd.DataFrame, p1: str, p2: str) -> str:
    """Generate Assistant-enhanced inventory comparison summary."""
    # Find available delta columns
    delta_cols = [col for col in pivot_df.columns if col.endswith('_delta')]
    value_delta = sum(pivot_df[col].sum() for col in delta_cols if 'value' in col.lower())
    
    summary_lines = [
        f"# Inventory Comparison ({p1} ➡ {p2})",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Δ Value:** {value_delta:,.0f}",
        f"**Items:** {len(pivot_df)}",
        "",
        "## Summary"
    ]
    
    if ASSISTANT_AVAILABLE and not pivot_df.empty:
        try:
            context = {
                "value_delta": value_delta,
                "periods": [p1, p2],
                "top_items": pivot_df.head(3).to_dict("records"),
                "delta_columns": delta_cols,
                "stats": {
                    "items_with_changes": len(pivot_df[pivot_df[delta_cols].sum(axis=1) != 0]) if delta_cols else 0,
                }
            }
            
            insights = assistants_answer(
                f"Analyze this inventory comparison from {p1} to {p2}. "
                f"Provide insights about inventory trends and working capital impact.",
                context=json.dumps(context, default=str)[:1500]
            )
            
            summary_lines.append(insights)
            
        except Exception:
            summary_lines.extend([
                f"- **Value change:** {value_delta:,.0f}",
                f"- **Items analyzed:** {len(pivot_df)}"
            ])
    else:
        summary_lines.extend([
            f"- **Value change:** {value_delta:,.0f}",
            f"- **Items analyzed:** {len(pivot_df)}"
        ])
    
    summary_lines.extend([
        "",
        "## Data Analysis",
        f"- **Delta columns:** {', '.join(delta_cols)}",
        f"- **Period comparison:** {p1} vs {p2}",
        "",
        "*Generated by SCIE Ethos Enterprise Comparison System*"
    ])
    
    return "\n".join(summary_lines)


def _generate_financial_summary(pivot_df: pd.DataFrame, p1: str, p2: str) -> str:
    """Generate Assistant-enhanced financial comparison summary."""
    value_delta = pivot_df['value_delta'].sum() if 'value_delta' in pivot_df.columns else 0
    
    summary_lines = [
        f"# Financial Comparison ({p1} ➡ {p2})",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Δ Value:** {value_delta:,.0f}",
        f"**Accounts:** {len(pivot_df)}",
        "",
        "## Analysis"
    ]
    
    if ASSISTANT_AVAILABLE and not pivot_df.empty:
        try:
            context = {
                "value_delta": value_delta,
                "periods": [p1, p2],
                "top_accounts": pivot_df.head(3).to_dict("records"),
                "stats": {
                    "accounts_increased": len(pivot_df[pivot_df.get('value_delta', 0) > 0]),
                    "accounts_decreased": len(pivot_df[pivot_df.get('value_delta', 0) < 0]),
                }
            }
            
            insights = assistants_answer(
                f"Analyze this financial comparison from {p1} to {p2}. "
                f"Highlight significant GL account changes and financial implications.",
                context=json.dumps(context, default=str)[:1500]
            )
            
            summary_lines.append(insights)
            
        except Exception:
            summary_lines.extend([
                f"- **Net change:** {value_delta:,.0f}",
                f"- **Accounts analyzed:** {len(pivot_df)}"
            ])
    else:
        summary_lines.extend([
            f"- **Net change:** {value_delta:,.0f}",
            f"- **Accounts analyzed:** {len(pivot_df)}"
        ])
    
    summary_lines.extend([
        "",
        "## Financial Impact",
        f"- **Period comparison:** {p1} vs {p2}",
        f"- **Net variance:** {value_delta:,.0f}",
        "",
        "*Generated by SCIE Ethos Enterprise Comparison System*"
    ])
    
    return "\n".join(summary_lines)

    return {"pivot": pivot, "summary_file": md_path, "excel_file": excel_path}


def _generate_wip_summary(pivot_df: pd.DataFrame, p1: str, p2: str, aging_columns: list) -> str:
    """Generate Assistant-enhanced WIP comparison summary."""
    total_delta = pivot_df['delta'].sum()
    
    summary_lines = [
        f"# WIP Aging Comparison ({p1} ➡ {p2})",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Δ:** {total_delta:,.0f}",
        f"**Records:** {len(pivot_df)}",
        "",
        "## Key Changes"
    ]
    
    if ASSISTANT_AVAILABLE and not pivot_df.empty:
        try:
            # Prepare context for Assistant analysis
            top_changes = pivot_df.head(5)
            context = {
                "total_delta": total_delta,
                "periods": [p1, p2],
                "aging_buckets": aging_columns,
                "top_changes": top_changes[['aging_bucket', 'delta']].to_dict("records") if 'aging_bucket' in top_changes.columns else [],
                "summary_stats": {
                    "positive_changes": len(pivot_df[pivot_df['delta'] > 0]),
                    "negative_changes": len(pivot_df[pivot_df['delta'] < 0]),
                    "max_increase": pivot_df['delta'].max(),
                    "max_decrease": pivot_df['delta'].min()
                }
            }
            
            insights = assistants_answer(
                f"Analyze this WIP aging comparison from {p1} to {p2}. "
                f"Provide 3 key insights about aging trends and business implications. "
                f"Focus on cash flow and operational risks.",
                context=json.dumps(context, default=str)[:2000]
            )
            
            summary_lines.append(insights)
            
        except Exception:
            # Fallback to basic analysis
            summary_lines.extend([
                f"- **Largest increase:** {pivot_df['delta'].max():,.0f}",
                f"- **Largest decrease:** {pivot_df['delta'].min():,.0f}",
                f"- **Average change:** {pivot_df['delta'].mean():,.0f}"
            ])
    else:
        # Basic summary without Assistant
        summary_lines.extend([
            f"- **Total change:** {total_delta:,.0f}",
            f"- **Records analyzed:** {len(pivot_df)}"
        ])
    
    summary_lines.extend([
        "",
        "## Aging Bucket Analysis",
        f"- **Buckets analyzed:** {', '.join(aging_columns)}",
        f"- **Period comparison:** {p1} vs {p2}",
        "",
        "*Generated by SCIE Ethos Enterprise Comparison System*"
    ])
    
    return "\n".join(summary_lines)
