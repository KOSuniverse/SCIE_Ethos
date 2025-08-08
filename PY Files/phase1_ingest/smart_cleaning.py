# smart_cleaning.py

import numpy as np
import pandas as pd
from logger import log_event
from metadata_utils import save_master_metadata_index

def fix_failed_int_columns(df, log):
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            if df[col].isnull().sum() > 0 or not np.allclose(df[col].dropna(), np.round(df[col].dropna())):
                df[col] = df[col].fillna(df[col].median())
                df[col] = np.round(df[col]).astype('Int64')
                log.append(f"Fixed '{col}' → filled NaNs + rounded to int.")
    return df, log

def cap_outliers(df, log, z_thresh=4):
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].dropna().size < 10: continue
        series = df[col]
        z_scores = (series - series.mean()) / series.std()
        outliers = np.abs(z_scores) > z_thresh
        if outliers.any():
            cap = series[~outliers].max()
            df[col] = np.where(outliers, cap, series)
            log.append(f"Capped outliers in '{col}' > {z_thresh} std dev.")
    return df, log

def drop_high_null_cols(df, log, threshold=0.9):
    for col in df.columns:
        null_frac = df[col].isnull().mean()
        if null_frac > threshold:
            df.drop(columns=[col], inplace=True)
            log.append(f"Dropped '{col}' with {null_frac:.0%} null values.")
    return df, log

def handle_special_values(df, log):
    special_values = set(['unknown', 'na', 'n/a', '-', '9999', 'none'])
    for col in df.columns:
        lower_vals = df[col].astype(str).str.lower().unique()
        found = set(lower_vals) & special_values
        if found:
            df[col] = df[col].replace(list(found), np.nan)
            log.append(f"Standardized placeholders in '{col}': {found}")
    return df, log

def smart_auto_fixer(df, log, actions=None):
    auto_fixes = {
        "fix_type_conversion": fix_failed_int_columns,
        "cap_outliers": cap_outliers,
        "drop_high_null_cols": drop_high_null_cols,
        "fix_placeholder_values": handle_special_values
    }

    if actions is None:
        # Default logic based on log keywords
        fixes_to_apply = []
        log_text = " ".join(log).lower()
        if 'int failed' in log_text or 'convert' in log_text:
            fixes_to_apply.append("fix_type_conversion")
        if 'outlier' in log_text:
            fixes_to_apply.append("cap_outliers")
        if 'missing values' in log_text:
            fixes_to_apply.append("drop_high_null_cols")
        if 'placeholder' in log_text:
            fixes_to_apply.append("fix_placeholder_values")
    else:
        fixes_to_apply = actions

    for fix in fixes_to_apply:
        if fix in auto_fixes:
            df, log = auto_fixes[fix](df, log)

    return df, log
