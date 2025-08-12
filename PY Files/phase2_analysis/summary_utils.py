# summary_utils.py

import os
import json
import datetime

# Enhanced foundation imports
try:
    from assistant_bridge import assistant_summarize
    ASSISTANT_AVAILABLE = True
except ImportError:
    ASSISTANT_AVAILABLE = False
    def assistant_summarize(content_type, data, context=""): return data

def generate_final_eda_summary(df, eda_text: str, paths: dict) -> str:
    """
    Generates a multi-section final EDA summary with inventory, E&O, cost allocation,
    subcontractor, financial, and data quality highlights.
    
    Enhanced to prefer assistant_summarize("final EDA", ...) with fallback to text as-is.

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

    raw_summary = "\n".join(summary)
    
    # Enhanced: Prefer assistant_summarize("final EDA", ...) with fallback to text as-is
    if ASSISTANT_AVAILABLE:
        try:
            # Prepare context for Assistant summarization
            context = f"""
            Dataset: {len(df)} rows, {len(df.columns)} columns
            Analysis type: Supply chain EDA summary
            Previous EDA text: {eda_text[:500]}...
            """
            
            # Use Assistant to enhance the summary
            enhanced_summary = assistant_summarize(
                content_type="final EDA",
                data=raw_summary,
                context=context
            )
            
            # Validate the enhanced summary
            if enhanced_summary and len(enhanced_summary) > len(raw_summary) * 0.5:
                full_summary = enhanced_summary
            else:
                # Fallback if Assistant response is too short or empty
                full_summary = raw_summary
                
        except Exception as e:
            print(f"Assistant summarization failed, using fallback: {e}")
            full_summary = raw_summary
    else:
        # Fallback to text as-is when Assistant not available
        full_summary = raw_summary

    # Save to JSON file with enhanced metadata
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "file": paths.get("filename", ""),
        "summary": full_summary,
        "raw_eda_text": eda_text,
        "assistant_enhanced": ASSISTANT_AVAILABLE,
        "data_stats": {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
    }

    exec_path = paths.get("summary_json", "")
    if exec_path:
        os.makedirs(os.path.dirname(exec_path), exist_ok=True)
        with open(exec_path, "w") as f:
            json.dump(record, f, indent=2)

    return full_summary

def create_assistant_enhanced_summary(df, analysis_type: str = "supply_chain", context: str = "") -> str:
    """
    Create a summary using Assistant enhancement with supply chain domain knowledge.
    """
    
    # Generate basic summary statistics
    basic_summary = f"""
    Dataset Overview:
    - Total Records: {len(df):,}
    - Total Columns: {len(df.columns)}
    - Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
    
    Data Quality:
    - Missing Data: {df.isnull().sum().sum():,} null values
    - Complete Records: {len(df.dropna()):,}
    
    Numeric Analysis:
    """
    
    # Add numeric column analysis
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    for col in numeric_cols[:5]:  # Limit to top 5 numeric columns
        basic_summary += f"\n    {col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}"
    
    # Add categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        basic_summary += f"\n\nCategorical Columns: {len(categorical_cols)}"
        for col in categorical_cols[:3]:  # Limit to top 3 categorical
            unique_count = df[col].nunique()
            basic_summary += f"\n    {col}: {unique_count} unique values"
    
    # Use Assistant enhancement if available
    if ASSISTANT_AVAILABLE:
        try:
            enhanced = assistant_summarize(
                content_type=f"{analysis_type} data analysis",
                data=basic_summary,
                context=context
            )
            return enhanced
        except Exception as e:
            print(f"Assistant enhancement failed: {e}")
    
    return basic_summary
