# eda_runner.py

import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enhanced foundation imports
try:
    from constants import DATA_ROOT
    from path_utils import canon_path
    from logger import log_event
except ImportError:
    # Fallback for standalone usage
    DATA_ROOT = "/Project_Root/04_Data"
    def canon_path(p): return str(Path(p).resolve())
    def log_event(msg, path=None): print(f"LOG: {msg}")

# Enhanced charting imports
try:
    from charting import save_bar_chart, save_scatter_plot, create_supply_chain_dashboard
except ImportError:
    # Fallback implementations
    def save_bar_chart(data, title, x_label, y_label, output_path, **kwargs):
        return output_path
    def save_scatter_plot(x, y, x_label, y_label, title, output_path, **kwargs):
        return output_path
    def create_supply_chain_dashboard(df, title, output_path):
        return output_path

def run_gpt_eda_actions(df: pd.DataFrame, actions: list, charts_folder: str = None, suffix: str = "") -> list:
    """
    Executes GPT-suggested EDA actions on a DataFrame.
    Enhanced to ensure charts saved to /04_Data/02_EDA_Charts with enterprise formatting.
    
    Args:
        df (pd.DataFrame): The cleaned dataset.
        actions (list): List of action dictionaries from GPT.
        charts_folder (str): Where to save generated charts (defaults to /04_Data/02_EDA_Charts).
        suffix (str): Optional suffix for file names.
    
    Returns:
        list: Paths to generated charts.
    """
    
    # Ensure charts are saved to correct enterprise folder
    if not charts_folder:
        charts_folder = f"{DATA_ROOT}/02_EDA_Charts"
    
    # Create folder if needed (enterprise requirement)
    charts_folder = canon_path(charts_folder)
    os.makedirs(charts_folder, exist_ok=True)
    
    chart_paths = []
    log_event(f"Starting EDA actions in folder: {charts_folder}")

    for i, action in enumerate(actions):
        act = action.get("action")
        
        try:
            if act == "histogram":
                col = action.get("column")
                if col and col in df.columns:
                    # Use enhanced charting with enterprise features
                    data = df[col].value_counts().head(20).to_dict()
                    title = f"Distribution: {col}"
                    output_path = os.path.join(charts_folder, f"hist_{col}{suffix}.png")
                    
                    path = save_bar_chart(
                        data=data,
                        title=title,
                        x_label=col,
                        y_label="Frequency",
                        output_path=output_path,
                        ai_enhance=True
                    )
                    chart_paths.append(path)

            elif act == "scatter":
                x, y = action.get("x"), action.get("y")
                if x and y and x in df.columns and y in df.columns:
                    # Filter out NaN values for better scatter plots
                    clean_df = df[[x, y]].dropna()
                    if len(clean_df) > 0:
                        title = f"Scatter: {x} vs {y}"
                        output_path = os.path.join(charts_folder, f"scatter_{x}_vs_{y}{suffix}.png")
                        
                        path = save_scatter_plot(
                            x=clean_df[x].values,
                            y=clean_df[y].values,
                            x_label=x,
                            y_label=y,
                            title=title,
                            output_path=output_path,
                            ai_enhance=True
                        )
                        chart_paths.append(path)

            elif act == "boxplot":
                col = action.get("column")
                if col and col in df.columns:
                    # Create boxplot with enterprise styling
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Use seaborn with better styling
                    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
                    sns.boxplot(data=df, y=col, ax=ax, color='#2E86AB')
                    
                    ax.set_title(f"Distribution Analysis: {col}", fontsize=14, fontweight='bold')
                    ax.set_ylabel(col, fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    path = os.path.join(charts_folder, f"box_{col}{suffix}.png")
                    plt.savefig(path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    chart_paths.append(path)

            elif act in ("correlation_matrix", "correlation_heatmap"):
                # Enhanced correlation heatmap
                numeric_df = df.select_dtypes(include=[float, int])
                if len(numeric_df.columns) > 1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
                    
                    corr = numeric_df.corr()
                    mask = corr.abs() < 0.1  # Hide weak correlations
                    
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                               square=True, ax=ax, cbar_kws={"shrink": .8},
                               mask=mask if len(corr) > 10 else None)
                    
                    ax.set_title("Correlation Analysis", fontsize=16, fontweight='bold', pad=20)
                    
                    path = os.path.join(charts_folder, f"correlation_heatmap{suffix}.png")
                    plt.savefig(path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    chart_paths.append(path)

            elif act == "groupby_topn":
                group = action.get("group")
                metric = action.get("metric")
                topn = action.get("topn", 10)
                if group and metric and group in df.columns and metric in df.columns:
                    # Calculate top N with better error handling
                    try:
                        result = df.groupby(group)[metric].sum().sort_values(ascending=False).head(topn)
                        
                        if len(result) > 0:
                            data = result.to_dict()
                            title = f"Top {topn} {group} by {metric}"
                            output_path = os.path.join(charts_folder, f"groupby_{group}_topn_{metric}{suffix}.png")
                            
                            path = save_bar_chart(
                                data=data,
                                title=title,
                                x_label=group,
                                y_label=metric,
                                output_path=output_path,
                                ai_enhance=True
                            )
                            chart_paths.append(path)
                    except Exception as groupby_error:
                        log_event(f"Groupby operation failed for {group}/{metric}: {groupby_error}")

            elif act == "supply_chain_dashboard":
                # Create comprehensive supply chain dashboard
                title = action.get("title", "Supply Chain Analysis Dashboard")
                output_path = os.path.join(charts_folder, f"supply_chain_dashboard{suffix}.png")
                
                path = create_supply_chain_dashboard(
                    df=df,
                    title=title,
                    output_path=output_path
                )
                chart_paths.append(path)

        except Exception as e:
            log_event(f"Failed to execute action {act}: {e}")
            print(f"⚠️ Failed to execute action {act}: {e}")

    log_event(f"Completed EDA actions: {len(chart_paths)} charts generated")
    return chart_paths

def create_enhanced_eda_summary(df: pd.DataFrame, chart_paths: list, charts_folder: str = None) -> dict:
    """
    Create enhanced EDA summary with enterprise insights and chart references.
    """
    
    if not charts_folder:
        charts_folder = f"{DATA_ROOT}/02_EDA_Charts"
    
    # Basic statistics
    summary = {
        "dataset_overview": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "missing_data_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
        },
        "column_analysis": {},
        "charts_generated": chart_paths,
        "charts_folder": charts_folder
    }
    
    # Analyze each column
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique())
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
                "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                "max": float(df[col].max()) if pd.notna(df[col].max()) else None
            })
        
        summary["column_analysis"][col] = col_info
    
    # Supply chain specific insights
    supply_chain_cols = {
        'wip': [col for col in df.columns if 'wip' in str(col).lower()],
        'cost': [col for col in df.columns if any(term in str(col).lower() for term in ['cost', 'value', 'price'])],
        'quantity': [col for col in df.columns if any(term in str(col).lower() for term in ['qty', 'quantity', 'count'])]
    }
    
    summary["supply_chain_insights"] = {}
    for category, cols in supply_chain_cols.items():
        if cols:
            summary["supply_chain_insights"][category] = {
                "columns_found": cols,
                "total_value": float(df[cols].sum().sum()) if cols else 0
            }
    
    return summary
