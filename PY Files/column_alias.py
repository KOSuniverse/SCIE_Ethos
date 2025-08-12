# column_alias.py

import json
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Import AI client for intelligent column mapping
try:
    from llm_client import get_openai_client, chat_completion
except ImportError:
    get_openai_client = None
    chat_completion = None

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

def build_reverse_alias_map(alias_group: dict) -> dict:
    """
    Enhanced reverse map builder with better handling.
    
    Args:
        alias_group: Dictionary mapping canonical names to variants
        
    Returns:
        dict: Reverse mapping from variant to canonical name
    """
    rev = {}
    for canonical, synonyms in (alias_group or {}).items():
        key = str(canonical).strip()
        rev[key.lower()] = key  # canonical maps to itself
        
        if isinstance(synonyms, (list, tuple, set)):
            for s in synonyms:
                rev[str(s).strip().lower()] = key
        elif synonyms is not None:
            rev[str(synonyms).strip().lower()] = key
    return rev

def remap_columns(df: pd.DataFrame, reverse_map: dict) -> pd.DataFrame:
    """
    Enhanced column remapping with case-insensitive matching.
    
    Args:
        df: DataFrame to remap
        reverse_map: Mapping from variant to canonical name
        
    Returns:
        DataFrame with remapped columns
    """
    if not reverse_map:
        return df.copy()
    
    # Case-insensitive lookup
    rev_lower = {str(k).strip().lower(): str(v).strip()
                 for k, v in reverse_map.items()}
    
    out = df.copy()
    new_columns = []
    
    for col in df.columns:
        col_clean = str(col).strip()
        col_lower = col_clean.lower()
        mapped_name = rev_lower.get(col_lower, col_clean)
        new_columns.append(mapped_name)
    
    out.columns = new_columns
    return out

# ================== AI-POWERED COLUMN INTELLIGENCE ==================

def ai_suggest_column_mappings(df: pd.DataFrame, existing_aliases: Dict = None, client=None) -> Dict:
    """
    Use AI to suggest intelligent column mappings based on data content and patterns.
    
    Args:
        df: DataFrame to analyze
        existing_aliases: Current alias mappings to consider
        client: OpenAI client
        
    Returns:
        dict: Suggested mappings and analysis
    """
    if client is None or chat_completion is None:
        return _fallback_column_suggestions(df, existing_aliases)
    
    try:
        # Prepare column analysis
        column_info = {}
        for col in df.columns[:20]:  # Limit to first 20 columns for token efficiency
            sample_values = df[col].dropna().head(10).astype(str).tolist()
            column_info[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": sample_values
            }
        
        # Prepare existing aliases context
        existing_context = ""
        if existing_aliases:
            canonical_names = list(existing_aliases.keys())[:10]  # Show some examples
            existing_context = f"Existing canonical column names include: {canonical_names}"
        
        prompt = f"""
        Analyze these columns from a supply chain dataset and suggest intelligent mappings to canonical names.
        
        {existing_context}
        
        Column Analysis:
        {json.dumps(column_info, indent=2)}
        
        For each column, suggest:
        1. Canonical name (standardized, snake_case)
        2. Confidence level (0.0-1.0)
        3. Reasoning
        4. Data quality observations
        
        Focus on supply chain terminology:
        - part_number, part_no, item_id for product identifiers
        - quantity, qty_on_hand, remaining_qty for quantities
        - unit_cost, extended_cost, inventory_value for monetary values
        - location, site, warehouse for locations
        - job_number, work_order for WIP tracking
        
        Return JSON:
        {{
            "suggested_mappings": {{
                "original_column": {{
                    "canonical_name": "part_number",
                    "confidence": 0.95,
                    "reasoning": "Contains part/product identifiers with consistent format",
                    "data_quality": "Good - no nulls, consistent format",
                    "transformations": ["strip_whitespace", "uppercase"]
                }}
            }},
            "new_canonical_names": ["any_new_canonical_names_not_in_existing"],
            "quality_issues": ["description of any data quality concerns"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in supply chain data standardization. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = chat_completion(client, messages, model="gpt-4o-mini")
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                return _fallback_column_suggestions(df, existing_aliases)
                
    except Exception as e:
        print(f"AI column mapping failed: {e}")
        return _fallback_column_suggestions(df, existing_aliases)

def ai_enhanced_alias_builder(df: pd.DataFrame, sheet_type: str = "unknown", 
                             existing_aliases: Dict = None) -> Tuple[Dict, List[str]]:
    """
    Build enhanced alias mappings using AI intelligence.
    
    Returns:
        Tuple[dict, List[str]]: (alias_mappings, log_messages)
    """
    log = []
    
    # Get AI client
    client = None
    if get_openai_client is not None:
        try:
            client = get_openai_client()
        except Exception as e:
            log.append(f"Could not initialize AI client for alias building: {e}")
    
    # Get AI suggestions
    ai_suggestions = ai_suggest_column_mappings(df, existing_aliases, client)
    
    # Build enhanced alias map
    enhanced_aliases = existing_aliases.copy() if existing_aliases else {}
    
    suggested_mappings = ai_suggestions.get('suggested_mappings', {})
    for original_col, mapping_info in suggested_mappings.items():
        canonical = mapping_info.get('canonical_name')
        confidence = mapping_info.get('confidence', 0.0)
        
        if canonical and confidence > 0.7:  # Only accept high-confidence suggestions
            if canonical not in enhanced_aliases:
                enhanced_aliases[canonical] = []
            
            # Add original column as variant if not already present
            if original_col not in enhanced_aliases[canonical]:
                enhanced_aliases[canonical].append(original_col)
                log.append(f"Added AI-suggested mapping: {original_col} → {canonical} (confidence: {confidence:.2f})")
    
    # Log quality issues
    for issue in ai_suggestions.get('quality_issues', []):
        log.append(f"AI detected data quality issue: {issue}")
    
    return enhanced_aliases, log

def smart_column_harmonization(dfs: List[pd.DataFrame], sheet_names: List[str] = None) -> Tuple[Dict, List[str]]:
    """
    Use AI to harmonize columns across multiple DataFrames intelligently.
    
    Args:
        dfs: List of DataFrames to harmonize
        sheet_names: Optional names for logging
        
    Returns:
        Tuple[dict, List[str]]: (unified_alias_map, log_messages)
    """
    log = []
    
    if not dfs:
        return {}, ["No DataFrames provided for harmonization"]
    
    # Collect all unique columns across sheets
    all_columns = set()
    column_examples = {}
    
    for i, df in enumerate(dfs):
        sheet_name = sheet_names[i] if sheet_names and i < len(sheet_names) else f"Sheet_{i+1}"
        for col in df.columns:
            all_columns.add(col)
            if col not in column_examples:
                # Store sample data for AI analysis
                sample_data = df[col].dropna().head(5).astype(str).tolist()
                column_examples[col] = {
                    "sheet": sheet_name,
                    "samples": sample_data,
                    "dtype": str(df[col].dtype),
                    "null_pct": float(df[col].isnull().mean())
                }
    
    log.append(f"Found {len(all_columns)} unique columns across {len(dfs)} sheets")
    
    # Get AI client
    client = None
    if get_openai_client is not None:
        try:
            client = get_openai_client()
        except Exception as e:
            log.append(f"Could not initialize AI client for harmonization: {e}")
            return _fallback_harmonization(all_columns, log)
    
    if client is None or chat_completion is None:
        return _fallback_harmonization(all_columns, log)
    
    try:
        prompt = f"""
        Harmonize these columns from multiple supply chain spreadsheet sheets into canonical mappings.
        
        Column Examples:
        {json.dumps(column_examples, indent=2)}
        
        Group similar columns under canonical names. For example:
        - "Part No", "Part Number", "Item ID" → "part_number"
        - "Qty", "Quantity", "On Hand Qty" → "quantity"
        - "Unit Cost", "Cost Each", "Price" → "unit_cost"
        
        Return JSON:
        {{
            "canonical_groups": {{
                "part_number": ["Part No", "Part Number", "Item ID"],
                "quantity": ["Qty", "Quantity", "On Hand Qty"],
                "unit_cost": ["Unit Cost", "Cost Each", "Price"]
            }},
            "ungrouped_columns": ["columns that don't fit clear patterns"],
            "confidence_notes": "explanation of grouping decisions"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at harmonizing supply chain data columns across multiple spreadsheets."},
            {"role": "user", "content": prompt}
        ]
        
        response = chat_completion(client, messages, model="gpt-4o-mini")
        
        try:
            result = json.loads(response)
            canonical_groups = result.get('canonical_groups', {})
            ungrouped = result.get('ungrouped_columns', [])
            
            log.append(f"AI grouped {sum(len(v) for v in canonical_groups.values())} columns into {len(canonical_groups)} canonical groups")
            log.append(f"Confidence notes: {result.get('confidence_notes', 'None')}")
            
            if ungrouped:
                log.append(f"Ungrouped columns: {ungrouped}")
            
            return canonical_groups, log
            
        except json.JSONDecodeError:
            log.append("Failed to parse AI harmonization response")
            return _fallback_harmonization(all_columns, log)
            
    except Exception as e:
        log.append(f"AI harmonization failed: {e}")
        return _fallback_harmonization(all_columns, log)

# ================== FALLBACK FUNCTIONS ==================

def _fallback_column_suggestions(df: pd.DataFrame, existing_aliases: Dict = None) -> Dict:
    """Fallback column suggestions when AI is not available."""
    suggestions = {}
    
    for col in df.columns:
        col_lower = col.lower()
        canonical = col.lower().replace(' ', '_').replace('-', '_')
        
        # Basic pattern matching
        if any(term in col_lower for term in ['part', 'item', 'product']):
            canonical = 'part_number'
        elif any(term in col_lower for term in ['qty', 'quantity']):
            canonical = 'quantity'
        elif any(term in col_lower for term in ['cost', 'price', 'value']):
            canonical = 'unit_cost'
        elif any(term in col_lower for term in ['location', 'site', 'warehouse']):
            canonical = 'location'
        
        suggestions[col] = {
            "canonical_name": canonical,
            "confidence": 0.6,
            "reasoning": "Pattern-based matching",
            "data_quality": "Not analyzed",
            "transformations": ["strip_whitespace"]
        }
    
    return {
        "suggested_mappings": suggestions,
        "new_canonical_names": [],
        "quality_issues": []
    }

def _fallback_harmonization(all_columns: set, log: List[str]) -> Tuple[Dict, List[str]]:
    """Fallback harmonization when AI is not available."""
    log.append("Using fallback pattern-based harmonization")
    
    groups = {}
    
    # Simple pattern-based grouping
    for col in all_columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['part', 'item']):
            groups.setdefault('part_number', []).append(col)
        elif any(term in col_lower for term in ['qty', 'quantity']):
            groups.setdefault('quantity', []).append(col)
        elif any(term in col_lower for term in ['cost', 'price', 'value']):
            groups.setdefault('unit_cost', []).append(col)
        else:
            # Create individual group for unmatched columns
            canonical = col.lower().replace(' ', '_').replace('-', '_')
            groups[canonical] = [col]
    
    log.append(f"Pattern-based grouping created {len(groups)} groups")
    return groups, log
