# matchmaker.py

import re
import difflib

def match_query_to_file_sheet(query, metadata_index):
    """
    Matches a user query to the most relevant file and sheet based on metadata.

    Args:
        query (str): Natural language question.
        metadata_index (dict): Loaded master metadata index.

    Returns:
        list of dicts: Matched files with metadata scores.
    """
    matches = []

    for entry in metadata_index.get("files", []):
        file_score = 0
        sheet_score = 0
        matched_sheets = []

        filename = entry.get("filename", "").lower()
        filepath = entry.get("path", "").lower()

        # Check filename relevance
        if any(word in filename for word in query.lower().split()):
            file_score += 1

        for sheet in entry.get("sheets", []):
            sheet_name = sheet.get("sheet_name", "").lower()
            sheet_type = sheet.get("sheet_type", "").lower()
            summary = sheet.get("summary", "").lower()
            columns = [c.lower() for c in sheet.get("columns", [])]

            match_strength = 0

            if any(word in sheet_name for word in query.lower().split()):
                match_strength += 1
            if sheet_type and sheet_type in query.lower():
                match_strength += 1
            if any(word in summary for word in query.lower().split()):
                match_strength += 1
            if any(col in query.lower() for col in columns):
                match_strength += 1

            if match_strength > 0:
                matched_sheets.append({
                    "sheet_name": sheet.get("sheet_name"),
                    "sheet_type": sheet_type,
                    "match_strength": match_strength
                })
                sheet_score += match_strength

        total_score = file_score + sheet_score

        if matched_sheets:
            matches.append({
                "filename": entry.get("filename"),
                "path": entry.get("path"),
                "score": total_score,
                "matched_sheets": matched_sheets
            })

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches
