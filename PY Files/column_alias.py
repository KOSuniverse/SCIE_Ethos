# column_alias.py

import json
import os

def load_alias_group(alias_path: str) -> dict:
    """
    Loads the global column alias mapping from JSON.

    Args:
        alias_path (str): Path to alias file.

    Returns:
        dict: { alias_name: [column_variants] }
    """
    if not os.path.exists(alias_path):
        return {}

    with open(alias_path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_column_alias(column_name: str, alias_map: dict) -> str:
    """
    Finds the standard alias for a given column name.

    Args:
        column_name (str): Raw column name.
        alias_map (dict): Alias mapping.

    Returns:
        str: Standard alias or original name.
    """
    column_name = column_name.strip().lower()
    for alias, variants in alias_map.items():
        if column_name in [v.lower() for v in variants]:
            return alias
    return column_name

def get_reverse_alias_map(alias_map: dict) -> dict:
    """
    Creates a reverse lookup map: variant -> alias.

    Returns:
        dict: Flattened column_name: alias
    """
    reverse = {}
    for alias, variants in alias_map.items():
        for v in variants:
            reverse[v.lower()] = alias
    return reverse
