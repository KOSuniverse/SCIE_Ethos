# comparison_utils.py

import os
import pandas as pd
from datetime import datetime

def compare_wip_aging(wip_dfs, output_folder):
    """
    Compares WIP aging buckets across two or more periods.
    Returns: dict with pivot table, summary file, and Excel file paths.
    """
    os.makedirs(output_folder, exist_ok=True)

    aging_columns = [
        '0_30_days', 'aging_31_60_days', 'aging_61_90_days',
        'aging_91_120_days', 'aging_121_150_days',
        'aging_151_180_days', 'over_180_days'
    ]

    melted_wip = pd.DataFrame()
    for df in wip_dfs:
        melted = df.melt(
            id_vars=['job_no', 'job_name', 'period', 'source_file'],
            value_vars=aging_columns,
            var_name='aging_bucket',
            value_name='value'
        )
        melted_wip = pd.concat([melted_wip, melted], ignore_index=True)

    periods = sorted(melted_wip['period'].unique())
    if len(periods) < 2:
        raise ValueError("At least two periods required.")
    p1, p2 = periods[0], periods[-1]

    pivot = melted_wip.pivot_table(
        index=['job_no', 'job_name', 'aging_bucket'],
        columns='period',
        values='value',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    pivot['delta'] = pivot[p2] - pivot[p1]
    pivot_sorted = pivot.sort_values(by='delta', ascending=False)

    file_prefix = f"wip_comparison_{p1.lower()}_to_{p2.lower()}"
    excel_path = os.path.join(output_folder, f"{file_prefix}.xlsx")
    pivot_sorted.to_excel(excel_path, index=False)

    summary_md = f"# WIP Comparison ({p1} ➡ {p2})\nTotal Δ: {pivot['delta'].sum():,.0f}"
    md_path = os.path.join(output_folder, f"{file_prefix}_summary.md")
    with open(md_path, "w") as f:
        f.write(summary_md)

    return {"pivot": pivot_sorted, "summary_file": md_path, "excel_file": excel_path}

def compare_inventory(inventory_dfs, output_folder):
    """
    Compares inventory files across periods, computing deltas in quantity and value.
    """
    os.makedirs(output_folder, exist_ok=True)

    df_all = pd.concat(inventory_dfs, ignore_index=True)
    periods = sorted(df_all['period'].unique())
    if len(periods) < 2:
        raise ValueError("Need at least two periods.")
    p1, p2 = periods[0], periods[-1]

    df_summary = df_all.groupby(['part_number', 'description', 'period'])[['remaining_quantity', 'value']].sum().reset_index()
    pivot = df_summary.pivot_table(
        index=['part_number', 'description'],
        columns='period',
        values=['remaining_quantity', 'value'],
        fill_value=0
    )
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
    pivot = pivot.reset_index()

    pivot['quantity_delta'] = pivot[f'remaining_quantity_{p2}'] - pivot[f'remaining_quantity_{p1}']
    pivot['value_delta'] = pivot[f'value_{p2}'] - pivot[f'value_{p1}']

    file_prefix = f"inventory_comparison_{p1.lower()}_to_{p2.lower()}"
    excel_path = os.path.join(output_folder, f"{file_prefix}.xlsx")
    pivot.to_excel(excel_path, index=False)

    summary_md = f"# Inventory Comparison ({p1} ➡ {p2})\nΔ Value: {pivot['value_delta'].sum():,.0f}"
    md_path = os.path.join(output_folder, f"{file_prefix}_summary.md")
    with open(md_path, "w") as f:
        f.write(summary_md)

    return {"pivot": pivot, "summary_file": md_path, "excel_file": excel_path}

def compare_financials(financial_dfs, output_folder):
    """
    Compares financial data (GL accounts) between two periods.
    """
    os.makedirs(output_folder, exist_ok=True)

    df_all = pd.concat(financial_dfs, ignore_index=True)
    periods = sorted(df_all['period'].unique())
    if len(periods) < 2:
        raise ValueError("Need at least two periods.")
    p1, p2 = periods[0], periods[-1]

    summary = df_all.groupby(['gl_account', 'period'])[['value']].sum().reset_index()
    pivot = summary.pivot_table(
        index='gl_account',
        columns='period',
        values='value',
        fill_value=0
    ).reset_index()

    pivot['value_delta'] = pivot[p2] - pivot[p1]

    file_prefix = f"financial_comparison_{p1.lower()}_to_{p2.lower()}"
    excel_path = os.path.join(output_folder, f"{file_prefix}.xlsx")
    pivot.to_excel(excel_path, index=False)

    summary_md = f"# Financial Comparison ({p1} ➡ {p2})\nΔ Value: {pivot['value_delta'].sum():,.0f}"
    md_path = os.path.join(output_folder, f"{file_prefix}_summary.md")
    with open(md_path, "w") as f:
        f.write(summary_md)

    return {"pivot": pivot, "summary_file": md_path, "excel_file": excel_path}
