# eda_runner.py

import json
from charting import save_bar_chart, save_scatter_plot
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_gpt_eda_actions(df, actions: list, charts_folder: str, suffix: str = "") -> list:
    """
    Executes GPT-suggested EDA actions on a DataFrame.
    
    Args:
        df (pd.DataFrame): The cleaned dataset.
        actions (list): List of action dictionaries from GPT.
        charts_folder (str): Where to save generated charts.
        suffix (str): Optional suffix for file names.
    
    Returns:
        list: Paths to generated charts.
    """
    os.makedirs(charts_folder, exist_ok=True)
    chart_paths = []

    for action in actions:
        act = action.get("action")
        
        try:
            if act == "histogram":
                col = action.get("column")
                if col in df.columns:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
                    ax.set_title(f"Histogram: {col}")
                    path = os.path.join(charts_folder, f"hist_{col}{suffix}.png")
                    plt.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    chart_paths.append(path)

            elif act == "scatter":
                x, y = action.get("x"), action.get("y")
                if x in df.columns and y in df.columns:
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=x, y=y, ax=ax)
                    ax.set_title(f"Scatter: {x} vs {y}")
                    path = os.path.join(charts_folder, f"scatter_{x}_vs_{y}{suffix}.png")
                    plt.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    chart_paths.append(path)

            elif act == "boxplot":
                col = action.get("column")
                if col in df.columns:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col], ax=ax)
                    ax.set_title(f"Boxplot: {col}")
                    path = os.path.join(charts_folder, f"box_{col}{suffix}.png")
                    plt.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    chart_paths.append(path)

            elif act in ("correlation_matrix", "correlation_heatmap"):
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = df.corr(numeric_only=True)
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                path = os.path.join(charts_folder, f"correlation_heatmap{suffix}.png")
                plt.savefig(path, bbox_inches="tight")
                plt.close(fig)
                chart_paths.append(path)

            elif act == "groupby_topn":
                group = action.get("group")
                metric = action.get("metric")
                topn = action.get("topn", 10)
                if group in df.columns and metric in df.columns:
                    result = df.groupby(group)[metric].sum().sort_values(ascending=False).head(topn)
                    fig, ax = plt.subplots()
                    result.plot(kind="bar", ax=ax)
                    ax.set_title(f"Top {topn} {group} by {metric}")
                    path = os.path.join(charts_folder, f"groupby_{group}_topn_{metric}{suffix}.png")
                    plt.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    chart_paths.append(path)

        except Exception as e:
            print(f"⚠️ Failed to execute action {act}: {e}")

    return chart_paths
