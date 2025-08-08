# summary_utils.py

import os
import json
import datetime

def generate_final_eda_summary(df, eda_text: str, paths: dict) -> str:
    """
    Generates a multi-section final EDA summary with inventory, E&O, cost allocation,
    subcontractor, financial, and data quality highlights.

    Args:
        df (pd.DataFrame): Cleaned dataset for a sheet_type.
        eda_text (str): Previously generated EDA text summary.
        paths (dict): Dictionary of project paths (see get_project_paths in file_utils).

    Returns:
        str: Full final summary text.
    """

    def format_currency(value):
        return f"${value:,.0f}"

    summary = []

    # 1. Inventory Value by Sheet
    if 'source_sheet' in df.columns and 'extended_cost' in df.columns:
        sheet_totals = df.groupby('source_sheet')['extended_cost'].sum()
        summary.append("📦 Inventory Value by Sheet:")
        summary.extend([f"  • {sheet}: {format_currency(v)}" for sheet, v in sheet_totals.items()])
        top_sheet = sheet_totals.idxmax()
        if sheet_totals.max() > 1_000_000:
            summary.append(f"⚠️ High value concentrated in '{top_sheet}'. Investigate aging or usage.")

    # 2. Estimated E&O Inventory
    if 'usage_to_stock' in df.columns and 'extended_cost' in df.columns:
        low_ratio_df = df[df['usage_to_stock'] < 0.1]
        eo_total = low_ratio_df['extended_cost'].sum()
        top_parts = low_ratio_df['part_no'].value_counts().head(3).to_dict() if 'part_no' in df.columns else {}
        summary.append(f"\n🚨 Estimated E&O (usage_to_stock < 0.1): {format_currency(eo_total)}")
        for part, count in top_parts.items():
            summary.append(f"  • Part {part}: {count} entries flagged")
        if eo_total > 100_000:
            summary.append("🧠 Suggestion: Add forecast data to identify root drivers.")

    # 3. Cost Allocation
    if {'cost_allocated', 'cost_unallocated'}.issubset(df.columns):
        alloc = df['cost_allocated'].sum()
        unalloc = df['cost_unallocated'].sum()
        pct_unalloc = unalloc / (alloc + unalloc + 1e-9)
        summary.extend([
            f"\n💰 Total Cost Allocated: {format_currency(alloc)}",
            f"💸 Unallocated Cost: {format_currency(unalloc)} ({pct_unalloc:.1%})"
        ])
        if pct_unalloc > 0.25:
            summary.append("⚠️ High unallocated cost — review allocation rules.")

    # 4. Subcontractor Late Costs
    if 'sub_con_late_costs' in df.columns:
        late_total = df['sub_con_late_costs'].sum()
        summary.append(f"\n⏱️ Subcontractor Late Costs: {format_currency(late_total)}")
        if late_total > 10_000:
            summary.append("📉 Action: Review supplier performance.")

    # 5. Financial Totals
    for field in ['labour', 'burden', 'ap_invoices']:
        if field in df.columns:
            summary.append(f"📊 {field.replace('_', ' ').title()}: {format_currency(df[field].sum())}")

    # 6. Missing Data Overview
    top_nulls = df.isnull().mean().sort_values(ascending=False).head()
    summary.append("\n🔍 Top Columns with Missing Data:")
    summary.extend([f"  • {col}: {frac:.1%} missing" for col, frac in top_nulls.items()])

    # 7. Misformatted Date Warning
    if 'last_used' in df.columns:
        numeric_like = df['last_used'].astype(str).str.match(r'^\d+$').sum()
        if numeric_like > 0:
            summary.append(f"\n📅 {numeric_like} entries in 'last_used' appear misformatted.")

    # 8. Final Guidance
    summary.append("\n✅ Summary generated from latest EDA actions.")
    summary.append("📈 Recommended next step: Cross-file or cross-period comparison.")

    full_summary = "\n".join(summary)

    # Save to JSON file
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "file": paths.get("filename", ""),
        "summary": full_summary,
        "raw_eda_text": eda_text
    }

    exec_path = paths.get("summary_json", "")
    if exec_path:
        os.makedirs(os.path.dirname(exec_path), exist_ok=True)
        with open(exec_path, "w") as f:
            json.dump(record, f, indent=2)

    return full_summary
