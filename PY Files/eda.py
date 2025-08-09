# eda.py

import pandas as pd
import numpy as np  

def generate_eda_summary(df: pd.DataFrame) -> dict:
    """
    Generates a summary of EDA results from the given DataFrame.

    Args:
        df (pd.DataFrame): The data to analyze.

    Returns:
        dict: Summary with key stats, correlations, nulls, and top movers.
    """
    summary = {}

    # Basic shape
    summary["num_rows"] = df.shape[0]
    summary["num_columns"] = df.shape[1]

    # Missing values
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    summary["missing_values"] = null_counts[null_counts > 0].to_dict()
    summary["missing_percent"] = null_percent[null_percent > 0].round(1).to_dict()

    # Correlation (numerics only)
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        corr_matrix = numeric_cols.corr().abs()
        upper_triangle = corr_matrix.where(
            ~np.tril(np.ones(corr_matrix.shape)).astype(bool))
        correlated_pairs = (
            upper_triangle.stack()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )
        summary["top_correlations"] = {
            f"{a} & {b}": round(v, 3) for (a, b), v in correlated_pairs.items()
        }

    # Descriptive stats
    if not numeric_cols.empty:
        stats = numeric_cols.describe().T[["mean", "std", "min", "max"]]
        summary["descriptive_stats"] = stats.round(2).to_dict()

    return summary
