# --- utils/column_mapping.py ---

import json
import streamlit as st
from collections import defaultdict
from openai import OpenAI
from utils.retry import openai_with_retry

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
CHAT_MODEL = "gpt-4o"

def detect_alias_conflicts(mapping):
    """Detect conflicts where multiple columns map to the same alias."""
    reverse = defaultdict(list)
    for col, alias in mapping.items():
        reverse[alias].append(col)
    conflicts = {
        alias: cols for alias, cols in reverse.items() 
        if len(cols) > 1 and alias != "ignore"
    }
    return conflicts

def map_columns_to_concepts(columns, global_aliases=None, preview=True):
    """
    Map Excel column headers to standardized business concepts using LLM.
    
    Args:
        columns: List of column names to map
        global_aliases: Existing global alias mappings
        preview: Whether to show preview/editing interface
    
    Returns:
        Dictionary mapping column names to concepts
    """
    if not columns:
        return {}
    
    unmapped = [col for col in columns if not global_aliases or col not in global_aliases]
    mapping = global_aliases.copy() if global_aliases else {}
    new_mapping = {}
    
    if unmapped:
        prompt = (
            "You are a data standardization assistant that helps map inconsistent Excel column names "
            "to a small set of standardized business concepts. Your job is to return a JSON dictionary "
            "where each key is the original column name and each value is the mapped concept name.\n\n"
            "📌 Mapping Instructions:\n"
            "- Map each original column header to a common business concept such as:\n"
            "  'part_number', 'quantity', 'description', 'location', 'date', 'value', 'status', etc.\n"
            "- If a column header is unclear, irrelevant, or junk (e.g., 'Sheet1', 'Unnamed', etc.), map it to 'ignore'.\n"
            "- Do not guess. If unsure, map to 'ignore'.\n\n"
            "🧾 Output Format:\n"
            "Return only a JSON dictionary, like this:\n"
            "{\n"
            '  "QTY": "quantity",\n'
            '  "part no": "part_number",\n'
            '  "xyz123": "ignore"\n'
            "}\n\n"
            f"🎯 Columns to map:\n{unmapped}\n\n"
            "Only return valid JSON. Do not include explanations, comments, or markdown formatting."
        )
        
        try:
            response = openai_with_retry(
                lambda: client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
            )
            raw = response.choices[0].message.content.strip()
            
            # Remove markdown wrapper if present
            if raw.startswith("```json") or raw.startswith("```"):
                raw = raw.strip("`").replace("json", "").strip()
            
            # Try to parse JSON
            try:
                new_mapping = json.loads(raw)
            except Exception:
                # Fallback: try to parse as key-value pairs
                new_mapping = {}
                for line in raw.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        new_mapping[k.strip().strip('"')] = v.strip().strip('"').rstrip(',')
                if not new_mapping:
                    st.warning("Could not parse column mapping from LLM output.")
            
            mapping.update(new_mapping)
            
        except Exception as e:
            st.warning(f"Column mapping failed: {e}")
    
    # Detect alias conflicts
    conflicts = detect_alias_conflicts(mapping)
    if conflicts:
        st.warning(f"Alias conflicts detected: {conflicts}")
    
    # Preview/audit in Streamlit
    if preview and new_mapping:
        st.write("Column mapping preview (edit if needed):")
        editable_json = st.text_area(
            "Edit mapping as JSON if needed:",
            value=json.dumps(mapping, indent=2),
            key="column_mapping_preview"
        )
        try:
            mapping = json.loads(editable_json)
        except Exception:
            st.error("Invalid JSON in edited mapping. Using previous mapping.")
    
    return mapping

def fuzzy_match_score(word, text, threshold=0.85):
    """Calculate fuzzy match score for keyword matching."""
    from difflib import SequenceMatcher
    return any(
        SequenceMatcher(None, word, token).ratio() >= threshold 
        for token in text.split()
    )
