# smart_cleaning.py

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional

# Import logging and AI client
try:
    from logger import log_event
    from llm_client import get_openai_client, chat_completion
except ImportError:
    def log_event(msg, path=None): print(f"LOG: {msg}")
    get_openai_client = None
    chat_completion = None

try:
    from metadata_utils import save_master_metadata_index
except ImportError:
    def save_master_metadata_index(data, path): pass

# ================== AI-POWERED DATA CLEANING ==================

def ai_analyze_data_quality(df: pd.DataFrame, sheet_type: str = "unknown", client=None) -> Dict:
    """
    Use AI to analyze data quality issues and suggest cleaning strategies.
    
    Returns:
        dict: Analysis results with recommendations
    """
    if client is None or chat_completion is None:
        return _fallback_data_analysis(df, sheet_type)
    
    try:
        # Prepare data sample for AI analysis
        sample_info = _prepare_data_sample(df)
        
        prompt = f"""
        You are a data quality expert analyzing a {sheet_type} dataset.
        
        Dataset Info:
        - Rows: {len(df)}
        - Columns: {len(df.columns)}
        - Column Details: {json.dumps(sample_info['column_analysis'], indent=2)}
        - Data Sample: {json.dumps(sample_info['sample_data'], indent=2)}
        
        Analyze the data quality and provide specific cleaning recommendations.
        
        Return JSON in this exact format:
        {{
            "quality_score": 0.85,
            "issues_found": [
                {{"type": "missing_values", "severity": "medium", "columns": ["col1"], "description": "..."}}
            ],
            "recommendations": [
                {{"action": "fill_missing", "columns": ["col1"], "method": "median", "reasoning": "..."}}
            ],
            "data_type_suggestions": [
                {{"column": "part_no", "current_type": "object", "suggested_type": "string", "reasoning": "..."}}
            ]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a data quality expert specializing in supply chain datasets. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = chat_completion(client, messages, model="gpt-4o-mini")
        
        # Parse AI response
        try:
            analysis = json.loads(response)
            return analysis
        except json.JSONDecodeError:
            # Extract JSON from response if wrapped in text
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                return _fallback_data_analysis(df, sheet_type)
                
    except Exception as e:
        log_event(f"AI data quality analysis failed: {e}")
        return _fallback_data_analysis(df, sheet_type)

def ai_smart_column_detection(df: pd.DataFrame, client=None) -> Dict:
    """
    Use AI to intelligently detect column meanings and suggest standardization.
    """
    if client is None or chat_completion is None:
        return _fallback_column_detection(df)
    
    try:
        # Sample data for AI analysis
        col_samples = {}
        for col in df.columns:
            sample_vals = df[col].dropna().head(10).astype(str).tolist()
            col_samples[col] = sample_vals
        
        prompt = f"""
        Analyze these columns from a supply chain dataset and identify their semantic meaning:
        
        Column Samples: {json.dumps(col_samples, indent=2)}
        
        For each column, determine:
        1. Semantic type (part_number, quantity, value, date, location, etc.)
        2. Data quality issues
        3. Standardization suggestions
        
        Return JSON:
        {{
            "column_mappings": {{
                "original_col_name": {{
                    "semantic_type": "part_number",
                    "confidence": 0.95,
                    "suggested_name": "part_no",
                    "issues": ["contains_nulls", "mixed_case"],
                    "transformations": ["strip_whitespace", "uppercase"]
                }}
            }}
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at analyzing supply chain data columns and identifying their semantic meaning."},
            {"role": "user", "content": prompt}
        ]
        
        response = chat_completion(client, messages, model="gpt-4o-mini")
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                return _fallback_column_detection(df)
                
    except Exception as e:
        log_event(f"AI column detection failed: {e}")
        return _fallback_column_detection(df)

def ai_guided_cleaning(df: pd.DataFrame, sheet_type: str = "unknown", log: List = None) -> Tuple[pd.DataFrame, List]:
    """
    AI-powered data cleaning that makes intelligent decisions based on data analysis.
    """
    if log is None:
        log = []
    
    # Get AI client
    client = None
    if get_openai_client is not None:
        try:
            client = get_openai_client()
        except Exception as e:
            log.append(f"Could not initialize AI client: {e}")
    
    # 1. AI Data Quality Analysis
    quality_analysis = ai_analyze_data_quality(df, sheet_type, client)
    log.append(f"AI Quality Score: {quality_analysis.get('quality_score', 'N/A')}")
    
    # 2. AI Column Detection
    column_analysis = ai_smart_column_detection(df, client)
    
    # 3. Apply AI-recommended cleaning
    cleaned_df = df.copy()
    
    for recommendation in quality_analysis.get('recommendations', []):
        try:
            cleaned_df, log = _apply_cleaning_recommendation(cleaned_df, recommendation, log)
        except Exception as e:
            log.append(f"Failed to apply recommendation {recommendation.get('action')}: {e}")
    
    # 4. Apply column transformations
    for col, mapping in column_analysis.get('column_mappings', {}).items():
        if col in cleaned_df.columns:
            try:
                cleaned_df, log = _apply_column_transformations(cleaned_df, col, mapping, log)
            except Exception as e:
                log.append(f"Failed to transform column {col}: {e}")
    
    # 5. Final validation
    final_issues = _validate_cleaned_data(cleaned_df, quality_analysis)
    if final_issues:
        log.extend([f"Validation issue: {issue}" for issue in final_issues])
    
    return cleaned_df, log

# ================== HELPER FUNCTIONS ==================

def _prepare_data_sample(df: pd.DataFrame, max_rows: int = 20) -> Dict:
    """Prepare data sample for AI analysis."""
    column_analysis = {}
    for col in df.columns:
        column_analysis[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": float(df[col].isnull().mean()),
            "unique_count": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).astype(str).tolist()
        }
        
        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(df[col]):
            column_analysis[col].update({
                "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                "std": float(df[col].std()) if not df[col].isnull().all() else None,
                "min": float(df[col].min()) if not df[col].isnull().all() else None,
                "max": float(df[col].max()) if not df[col].isnull().all() else None
            })
    
    sample_data = df.head(max_rows).to_dict(orient='records')
    
    return {
        "column_analysis": column_analysis,
        "sample_data": sample_data
    }

def _apply_cleaning_recommendation(df: pd.DataFrame, recommendation: Dict, log: List) -> Tuple[pd.DataFrame, List]:
    """Apply a specific cleaning recommendation."""
    action = recommendation.get('action')
    columns = recommendation.get('columns', [])
    method = recommendation.get('method')
    
    if action == 'fill_missing':
        for col in columns:
            if col in df.columns:
                if method == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    log.append(f"Filled missing values in '{col}' with median")
                elif method == 'mode':
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                    log.append(f"Filled missing values in '{col}' with mode: {mode_val}")
                elif method == 'forward_fill':
                    df[col] = df[col].fillna(method='ffill')
                    log.append(f"Forward filled missing values in '{col}'")
    
    elif action == 'remove_outliers':
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers_removed = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                log.append(f"Removed {outliers_removed} outliers from '{col}'")
    
    elif action == 'standardize_format':
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                log.append(f"Standardized format for '{col}' (uppercase, trimmed)")
    
    return df, log

def _apply_column_transformations(df: pd.DataFrame, col: str, mapping: Dict, log: List) -> Tuple[pd.DataFrame, List]:
    """Apply transformations to a specific column."""
    transformations = mapping.get('transformations', [])
    
    for transform in transformations:
        if transform == 'strip_whitespace':
            df[col] = df[col].astype(str).str.strip()
            log.append(f"Stripped whitespace from '{col}'")
        elif transform == 'uppercase':
            df[col] = df[col].astype(str).str.upper()
            log.append(f"Converted '{col}' to uppercase")
        elif transform == 'remove_special_chars':
            df[col] = df[col].astype(str).str.replace(r'[^\w\s-]', '', regex=True)
            log.append(f"Removed special characters from '{col}'")
    
    return df, log

def _validate_cleaned_data(df: pd.DataFrame, quality_analysis: Dict) -> List[str]:
    """Validate the cleaned data and return any remaining issues."""
    issues = []
    
    # Check if critical issues were resolved
    for issue in quality_analysis.get('issues_found', []):
        if issue.get('severity') == 'high':
            issue_type = issue.get('type')
            columns = issue.get('columns', [])
            
            if issue_type == 'missing_values':
                for col in columns:
                    if col in df.columns and df[col].isnull().any():
                        issues.append(f"High severity missing values still present in {col}")
    
    return issues

# ================== FALLBACK FUNCTIONS ==================

def _fallback_data_analysis(df: pd.DataFrame, sheet_type: str) -> Dict:
    """Fallback analysis when AI is not available."""
    issues = []
    recommendations = []
    
    # Basic missing value analysis
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0.5:
            issues.append({
                "type": "missing_values",
                "severity": "high",
                "columns": [col],
                "description": f"Column {col} has {null_pct:.1%} missing values"
            })
            recommendations.append({
                "action": "fill_missing",
                "columns": [col],
                "method": "mode" if not pd.api.types.is_numeric_dtype(df[col]) else "median",
                "reasoning": "High missing value percentage"
            })
    
    return {
        "quality_score": 0.7,
        "issues_found": issues,
        "recommendations": recommendations,
        "data_type_suggestions": []
    }

def _fallback_column_detection(df: pd.DataFrame) -> Dict:
    """Fallback column detection when AI is not available."""
    mappings = {}
    
    for col in df.columns:
        col_lower = col.lower()
        semantic_type = "unknown"
        
        if any(term in col_lower for term in ['part', 'item', 'product']):
            semantic_type = "part_identifier"
        elif any(term in col_lower for term in ['qty', 'quantity', 'count']):
            semantic_type = "quantity"
        elif any(term in col_lower for term in ['value', 'cost', 'price']):
            semantic_type = "monetary_value"
        elif any(term in col_lower for term in ['date', 'time']):
            semantic_type = "date_time"
        
        mappings[col] = {
            "semantic_type": semantic_type,
            "confidence": 0.6,
            "suggested_name": col.lower().replace(' ', '_'),
            "issues": [],
            "transformations": ["strip_whitespace"]
        }
    
    return {"column_mappings": mappings}

# ================== LEGACY COMPATIBILITY ==================

# Keep existing functions for backward compatibility
def fix_failed_int_columns(df, log):
    """Legacy function - now enhanced with AI guidance."""
    return ai_guided_cleaning(df, "unknown", log)

def cap_outliers(df, log, z_thresh=4):
    """Legacy function maintained for compatibility."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].dropna().size < 10: 
            continue
        series = df[col]
        z_scores = (series - series.mean()) / series.std()
        outliers = np.abs(z_scores) > z_thresh
        if outliers.any():
            cap = series[~outliers].max()
            df[col] = np.where(outliers, cap, series)
            log.append(f"Capped outliers in '{col}' > {z_thresh} std dev.")
    return df, log

def drop_high_null_cols(df, log, threshold=0.9):
    """Legacy function maintained for compatibility."""
    for col in df.columns:
        null_frac = df[col].isnull().mean()
        if null_frac > threshold:
            df.drop(columns=[col], inplace=True)
            log.append(f"Dropped '{col}' with {null_frac:.0%} null values.")
    return df, log

def handle_special_values(df, log):
    """Legacy function maintained for compatibility."""
    special_values = set(['unknown', 'na', 'n/a', '-', '9999', 'none'])
    for col in df.columns:
        lower_vals = df[col].astype(str).str.lower().unique()
        found = set(lower_vals) & special_values
        if found:
            df[col] = df[col].replace(list(found), np.nan)
            log.append(f"Standardized placeholders in '{col}': {found}")
    return df, log

def smart_auto_fixer(df, log, actions=None):
    """
    Enhanced smart auto fixer - now AI-powered by default.
    Falls back to rule-based approach if AI is unavailable.
    """
    # Try AI-guided cleaning first
    try:
        client = get_openai_client() if get_openai_client else None
        if client:
            return ai_guided_cleaning(df, "unknown", log)
    except Exception as e:
        log.append(f"AI cleaning failed, falling back to rule-based: {e}")
    
    # Fallback to original rule-based approach
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
