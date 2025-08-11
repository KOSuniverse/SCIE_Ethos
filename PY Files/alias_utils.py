
import json
import pandas as pd
import re

# PY Files/alias_utils.py
"""Legacy shim — canonical alias logic lives in column_alias.py."""
from warnings import warn as _warn
_warn("alias_utils.py is deprecated; use column_alias.py", stacklevel=1)
from column_alias import (  # re-export for backward compatibility
    load_alias_group,
    build_reverse_alias_map,
    normalize_columns,  # if present; safe to export even if unused
)
__all__ = ["load_alias_group", "build_reverse_alias_map", "normalize_columns"]


def clean_column(col):
    col = str(col)
    return re.sub(r'\W+', '_', col).strip().lower()

def load_alias_group(alias_path):
    with open(alias_path, 'r') as f:
        return json.load(f)

def build_reverse_alias_map(alias_group_dict):
    reverse_map = {}
    for std_name, props in alias_group_dict.items():
        for alias in props.get("aliases", []):
            reverse_map[alias.strip().lower()] = std_name
    return reverse_map

def remap_columns(df, reverse_map):
    col_mapping = {
        col: reverse_map[col.lower()]
        for col in df.columns if col.lower() in reverse_map
    }
    return df.rename(columns=col_mapping)

def alias_map_to_csv_v2(alias_group, csv_path):
    rows = []
    for std_name, props in alias_group.items():
        for alias in props.get("aliases", []):
            rows.append({
                "alias_column": alias,
                "standard_name": std_name,
                "category": props.get("category", ""),
                "erp_module": props.get("erp_module", "")
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"📄 Alias map exported to CSV: {csv_path}")

def export_json_to_csv(json_path, csv_path):
    with open(json_path, "r") as f:
        meta = json.load(f)
    df_dict = pd.DataFrame.from_dict(meta, orient="index").reset_index().rename(columns={"index": "standard_name"})
    df_dict.to_csv(csv_path, index=False)
    print(f"Exported JSON metadata to CSV: {csv_path}")

def import_csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Group aliases under standard_name
    grouped = df.groupby("standard_name")
    structured = {}
    for std_name, group in grouped:
        aliases = group["alias_column"].dropna().unique().tolist()
        category = group["category"].dropna().unique().tolist()
        erp_module = group["erp_module"].dropna().unique().tolist()
        structured[std_name] = {
            "aliases": aliases,
            "category": category[0] if category else "",
            "erp_module": erp_module[0] if erp_module else ""
        }
    with open(json_path, "w") as f:
        json.dump(structured, f, indent=2)
    print(f"Imported CSV data dictionary to JSON: {json_path}")
