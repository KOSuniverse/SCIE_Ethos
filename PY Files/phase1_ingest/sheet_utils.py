# sheet_utils.py

import os
import re
import json

DEFAULT_ALIAS_PATH = "Project_Root/04_Data/04_Metadata/sheet_aliases.json"

KNOWN_LOCATIONS = ['US', 'MX', 'CA', 'UK', 'EU', 'Thailand', 'Germany']
KNOWN_TYPES = ['WIP', 'Raw Materials', 'Raw', 'Finished Goods', 'FG', 'RM', 'Usage']
KNOWN_CONTEXT_TERMS = KNOWN_LOCATIONS + KNOWN_TYPES


def load_sheet_aliases(path=DEFAULT_ALIAS_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load sheet_aliases.json: {e}")
        return {}


def _feature_hint_from_columns(df) -> str:
    cols_lower = [str(c).lower() for c in df.columns]
    part_cols = [c for c in cols_lower if "part" in c and ("no" in c or "number" in c or "id" in c)]
    qty_cols  = [c for c in cols_lower if any(k in c for k in ["qty", "quantity", "on_hand"])]
    val_cols  = [c for c in cols_lower if any(k in c for k in ["value", "extended_cost", "inventory_value", "total_cost"])]
    job_cols  = [c for c in cols_lower if "job" in c or "work_order" in c or c in ("wo", "wo_no", "wo_number")]
    aging_cols = [c for c in cols_lower if "days" in c or "aging" in c]

    if part_cols and (qty_cols or val_cols):
        return "inventory"
    if job_cols and aging_cols:
        return "wip"
    return "unclassified"


def normalize_sheet_type(sheet_name, df, sheet_aliases):
    """
    Backward-compatible: returns a STRING type.
    Name-based alias first; column-based inference second.
    Inventory overrides WIP if both signals conflict.
    """
    sheet_name_clean = (sheet_name or "").strip().lower()
    name_hint = "unclassified"

    # 1) name-based alias
    for alias, norm in (sheet_aliases or {}).items():
        if alias.lower() in sheet_name_clean:
            name_hint = (norm or "unclassified").strip().lower()
            break

    # 2) column-based
    feature_hint = _feature_hint_from_columns(df)

    # decision: inventory beats WIP if name says WIP
    if name_hint == "wip" and feature_hint == "inventory":
        return "inventory"
    if name_hint != "unclassified":
        return name_hint
    return feature_hint


def classify_sheet(sheet_name, df, sheet_aliases):
    """
    Rich classifier for metadata. Returns dict with name_hint, feature_hint, final_type, type_resolution.
    """
    sheet_name_clean = (sheet_name or "").strip().lower()
    name_hint = "unclassified"

    for alias, norm in (sheet_aliases or {}).items():
        if alias.lower() in sheet_name_clean:
            name_hint = (norm or "unclassified").strip().lower()
            break

    feature_hint = _feature_hint_from_columns(df)

    if name_hint == "wip" and feature_hint == "inventory":
        final_type = "inventory"
        resolution = "feature_override_name"
    elif name_hint != "unclassified":
        final_type = name_hint
        resolution = "name_hint"
    else:
        final_type = feature_hint
        resolution = "feature_hint"

    return {
        "name_hint": name_hint,
        "feature_hint": feature_hint,
        "final_type": final_type,
        "type_resolution": resolution
    }


def extract_locations_and_context(filename, sheet_name, df):
    text_sources = {
        "file_name": filename,
        "sheet_name": sheet_name,
        "column_headers": " ".join(df.columns.astype(str)),
        "column_sample_values": " ".join(
            df.head(20).astype(str).fillna("").agg(" ".join, axis=1).tolist()
        )
    }

    found_tags = set()
    detected_from = set()

    for source_name, text in text_sources.items():
        for term in KNOWN_CONTEXT_TERMS:
            if re.search(rf'\b{re.escape(term)}\b', text, re.IGNORECASE):
                found_tags.add(term)
                detected_from.add(source_name)

    return {
        "locations": sorted(found_tags),
        "inferred_from": sorted(detected_from)
    }

