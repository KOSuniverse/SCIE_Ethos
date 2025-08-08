# ranking_utils.py

import pandas as pd

def rank_task(df: pd.DataFrame, entity_type: str, top_n: int = 10, filters: dict = None) -> pd.DataFrame:
    """
    Ranks comparison results based on value or delta columns.

    Args:
        df (pd.DataFrame): Pivoted comparison DataFrame.
        entity_type (str): 'wip', 'inventory', or 'finance'.
        top_n (int): Number of top results to return.
        filters (dict): Optional filter conditions {column: [values]}.

    Returns:
        pd.DataFrame: Ranked subset of the DataFrame.
    """
    if entity_type not in ['wip', 'inventory', 'finance']:
        raise ValueError("entity_type must be 'wip', 'inventory', or 'finance'.")

    df_filtered = df.copy()

    # Apply filters if provided
    if filters:
        for col, values in filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col].isin(values)]

    # Entity-specific configs
    if entity_type == 'wip':
        sort_col = 'delta'
        required_cols = ['job_no', 'job_name', 'aging_bucket', 'delta', 'direction']
    elif entity_type == 'inventory':
        sort_col = 'value_delta'
        required_cols = ['part_number', 'description', 'value_delta', 'direction']
    else:  # finance
        sort_col = 'value_delta'
        required_cols = ['gl_account', 'value_delta', 'direction']

    if sort_col not in df_filtered.columns:
        raise KeyError(f"Missing required sort column: {sort_col}")

    ranked = df_filtered.sort_values(by=sort_col, key=abs, ascending=False)

    missing_cols = [col for col in required_cols if col not in ranked.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    return ranked[required_cols].head(top_n)
