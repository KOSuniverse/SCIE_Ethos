# pipeline.py

import os
import json
import pandas as pd
from datetime import datetime

from column_alias import load_alias_group, build_reverse_alias_map, remap_columns
from metadata_utils import save_master_metadata_index
from summarizer import summarize_data_context
from eda import generate_eda_summary
from file_utils import list_cleaned_files
from constants import *

def clean_and_standardize_sheet(sheet_df, alias_path):
    alias_group = load_alias_group(alias_path)
    reverse_map = build_reverse_alias_map(alias_group)
    cleaned = remap_columns(sheet_df.copy(), reverse_map)
    return cleaned

def run_pipeline_on_file(xls_path, alias_path, output_prefix, output_folder):
    xls = pd.ExcelFile(xls_path)
    cleaned_sheets = []
    all_metadata = []

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            df = clean_and_standardize_sheet(df, alias_path)
            df['source_sheet'] = sheet

            summary_text = summarize_data_context(generate_eda_summary(df), client=None)
            metadata_entry = {
                "filename": os.path.basename(xls_path),
                "sheet_name": sheet,
                "columns": list(df.columns),
                "record_count": len(df),
                "sheet_type": "unclassified",
                "summary_text": summary_text
            }
            all_metadata.append(metadata_entry)
            cleaned_sheets.append(df)
        except Exception as e:
            print(f"⚠️ Failed to process sheet '{sheet}': {e}")

    # Save metadata
    metadata_path = os.path.join(output_folder, f"{output_prefix}_metadata.json")
    save_master_metadata_index({"files": all_metadata}, metadata_path)

    # Save cleaned file
    cleaned_file_path = os.path.join(output_folder, f"{output_prefix}_cleaned.xlsx")
    with pd.ExcelWriter(cleaned_file_path, engine='xlsxwriter') as writer:
        for df in cleaned_sheets:
            name = df['source_sheet'].iloc[0]
            df.to_excel(writer, sheet_name=name[:31], index=False)

    return cleaned_file_path, metadata_path
