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
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load sheet_aliases.json: {e}")
        return {}

def normalize_sheet_type(sheet_name, df, sheet_aliases):
    sheet_name_clean = sheet_name.strip().lower()
    inferred_type = "unclassified"

    for alias, norm in sheet_aliases.items():
        if alias.lower() in sheet_name_clean:
            inferred_type = norm
            break

    # Column-based overrides
    part_cols = [c for c in df.columns if "part" in c.lower() and "number" in c.lower()]
    job_cols = [c for c in df.columns if "job" in c.lower() and "name" in c.lower()]
    aging_cols = [c for c in df.columns if "days" in c.lower() or "aging" in c.lower()]

    if inferred_type == "wip" and part_cols:
        inferred_type = "inventory"

    if inferred_type == "unclassified":
        if part_cols:
            inferred_type = "inventory"
        elif job_cols and aging_cols:
            inferred_type = "wip"

    return inferred_type

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
