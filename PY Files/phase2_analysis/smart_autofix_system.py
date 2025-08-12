# smart_autofix_system.py
"""
Smart Auto-Fix System matching the original Colab workflow.
Provides GPT-driven data cleaning and issue correction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import re
import json

# LLM integration
try:
    from llm_client import get_openai_client, chat_completion
except ImportError:
    print("⚠️ LLM modules not available - auto-fix will be rule-based only")
    get_openai_client = None
    chat_completion = None

class SmartAutoFixSystem:
    """
    Intelligent auto-fix system that handles data quality issues.
    Uses both rule-based and AI-driven approaches for data cleaning.
    """
    
    def __init__(self):
        self.client = None
        self.cleaning_log = []
        
        if get_openai_client:
            try:
                self.client = get_openai_client()
            except Exception as e:
                print(f"⚠️ Could not initialize LLM client: {e}")
    
    def run_comprehensive_autofix(self, 
                                 df: pd.DataFrame, 
                                 sheet_name: str = "Sheet1",
                                 aggressive_mode: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        Run comprehensive auto-fix pipeline matching Colab workflow.
        
        Args:
            df: Input DataFrame
            sheet_name: Name of the sheet for logging
            aggressive_mode: Whether to apply more aggressive fixes
            
        Returns:
            Tuple of (cleaned_DataFrame, cleaning_operations_log)
        """
        
        print(f"🧹 Starting Comprehensive Auto-Fix for {sheet_name}")
        self.cleaning_log = []
        
        # Store original for comparison
        original_df = df.copy()
        
        # STEP 1: Type Inference & Correction
        print("  🔍 STEP 1: GPT Type Inference & Correction")
        df = self._gpt_type_inference(df, sheet_name)
        
        # STEP 2: Null Value Imputation
        print("  🔧 STEP 2: Intelligent Null Imputation")
        df = self._intelligent_null_imputation(df, sheet_name)
        
        # STEP 3: Duplicate Detection & Removal
        print("  🔄 STEP 3: Duplicate Detection & Handling")
        df = self._handle_duplicates(df, sheet_name, aggressive_mode)
        
        # STEP 4: Outlier Detection & Treatment
        print("  📊 STEP 4: Outlier Detection via Z-Score")
        df = self._outlier_detection_treatment(df, sheet_name, aggressive_mode)
        
        # STEP 5: ID Column Validation
        print("  🆔 STEP 5: ID Uniqueness Validation")
        df = self._validate_id_columns(df, sheet_name)
        
        # STEP 6: Categorical Value Cleaning
        print("  🏷️ STEP 6: Categorical Profiling & Cleaning")
        df = self._categorical_cleaning(df, sheet_name)
        
        # STEP 7: Text Column Processing
        print("  ✍️ STEP 7: Free Text & High-Uniqueness Processing")
        df = self._text_column_processing(df, sheet_name)
        
        # STEP 8: Placeholder Value Replacement
        print("  🧽 STEP 8: Replace Placeholders with NaN")
        df = self._replace_placeholders(df, sheet_name)
        
        # STEP 9: AI-Driven Final Cleanup (if available)
        if self.client:
            print("  🧠 STEP 9: AI-Driven Final Cleanup")
            df = self._ai_driven_cleanup(df, original_df, sheet_name)
        
        print(f"✅ Auto-Fix Complete! Applied {len(self.cleaning_log)} operations")
        return df, self.cleaning_log.copy()
    
    def _gpt_type_inference(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Use GPT to infer and correct column data types."""
        
        if not self.client:
            return self._rule_based_type_inference(df, sheet_name)
        
        try:
            # Analyze column patterns
            column_info = {}
            for col in df.columns:
                sample_values = df[col].dropna().head(10).astype(str).tolist()
                column_info[col] = {
                    "current_dtype": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique()),
                    "sample_values": sample_values
                }
            
            # Ask GPT for type recommendations
            context = f"""
Analyze these columns from dataset '{sheet_name}' and recommend optimal data types:

{json.dumps(column_info, indent=2)}

For each column, recommend the best pandas dtype from:
- int64, float64, datetime64, bool, category, object

Consider:
- Sample values and patterns
- Null counts and data completeness
- Business context (IDs, dates, categories, etc.)

Return JSON format:
{{"column_name": {{"recommended_type": "dtype", "reason": "explanation"}}}}
"""
            
            messages = [
                {"role": "system", "content": "You are a data engineering expert specializing in optimal data type selection for business datasets."},
                {"role": "user", "content": context}
            ]
            
            response = chat_completion(self.client, messages, model="gpt-4o-mini")
            
            # Parse recommendations
            try:
                recommendations = json.loads(response.strip())
                
                # Apply type conversions
                for col, rec in recommendations.items():
                    if col in df.columns:
                        recommended_type = rec.get("recommended_type")
                        reason = rec.get("reason", "GPT recommendation")
                        
                        try:
                            if recommended_type == "datetime64":
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                self.cleaning_log.append(f"Converted {col} to datetime: {reason}")
                            elif recommended_type == "category":
                                df[col] = df[col].astype('category')
                                self.cleaning_log.append(f"Converted {col} to category: {reason}")
                            elif recommended_type in ["int64", "float64"]:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                if recommended_type == "int64":
                                    # Only convert to int if no NaNs introduced
                                    if not df[col].isnull().any():
                                        df[col] = df[col].astype('int64')
                                self.cleaning_log.append(f"Converted {col} to numeric: {reason}")
                            elif recommended_type == "bool":
                                # Convert common boolean representations
                                bool_map = {
                                    'true': True, 'false': False, 'yes': True, 'no': False,
                                    '1': True, '0': False, 'y': True, 'n': False
                                }
                                df[col] = df[col].astype(str).str.lower().map(bool_map)
                                self.cleaning_log.append(f"Converted {col} to boolean: {reason}")
                                
                        except Exception as e:
                            self.cleaning_log.append(f"Failed to convert {col} to {recommended_type}: {e}")
                
            except json.JSONDecodeError:
                self.cleaning_log.append("GPT type inference response could not be parsed, using rule-based")
                return self._rule_based_type_inference(df, sheet_name)
                
        except Exception as e:
            self.cleaning_log.append(f"GPT type inference failed: {e}")
            return self._rule_based_type_inference(df, sheet_name)
        
        return df
    
    def _rule_based_type_inference(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Fallback rule-based type inference."""
        
        for col in df.columns:
            try:
                # Try numeric conversion
                if df[col].dtype == 'object':
                    # Check if it's numeric
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # Try to convert to numeric
                        numeric_series = pd.to_numeric(non_null_values, errors='coerce')
                        if numeric_series.notna().sum() / len(non_null_values) > 0.8:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            self.cleaning_log.append(f"Rule-based: Converted {col} to numeric")
                            continue
                        
                        # Check for datetime patterns
                        datetime_patterns = [
                            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                        ]
                        
                        sample_str = str(non_null_values.iloc[0])
                        if any(re.search(pattern, sample_str) for pattern in datetime_patterns):
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                self.cleaning_log.append(f"Rule-based: Converted {col} to datetime")
                                continue
                            except:
                                pass
                        
                        # Check for categorical data (low cardinality)
                        if df[col].nunique() / len(df) < 0.1 and df[col].nunique() < 50:
                            df[col] = df[col].astype('category')
                            self.cleaning_log.append(f"Rule-based: Converted {col} to category")
            
            except Exception as e:
                self.cleaning_log.append(f"Rule-based type inference failed for {col}: {e}")
        
        return df
    
    def _intelligent_null_imputation(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Intelligent null value imputation based on column types and patterns."""
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count == 0:
                continue
            
            null_percentage = (null_count / len(df)) * 100
            
            try:
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric imputation
                    if null_percentage < 5:
                        # Use median for small amounts of missing data
                        fill_value = df[col].median()
                        df[col].fillna(fill_value, inplace=True)
                        self.cleaning_log.append(f"Imputed {null_count} nulls in {col} with median ({fill_value:.2f})")
                    elif null_percentage < 20:
                        # Use mean for moderate missing data
                        fill_value = df[col].mean()
                        df[col].fillna(fill_value, inplace=True)
                        self.cleaning_log.append(f"Imputed {null_count} nulls in {col} with mean ({fill_value:.2f})")
                    else:
                        # High missing percentage - flag for review
                        self.cleaning_log.append(f"High null percentage in {col} ({null_percentage:.1f}%) - consider column removal")
                
                elif df[col].dtype == 'category' or (df[col].dtype == 'object' and df[col].nunique() < 20):
                    # Categorical imputation
                    if null_percentage < 10:
                        mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col].fillna(mode_value, inplace=True)
                        self.cleaning_log.append(f"Imputed {null_count} nulls in {col} with mode ({mode_value})")
                    else:
                        # Create "Missing" category
                        df[col].fillna('Missing', inplace=True)
                        self.cleaning_log.append(f"Imputed {null_count} nulls in {col} with 'Missing' category")
                
                elif df[col].dtype == 'datetime64[ns]':
                    # DateTime imputation - usually leave as NaN or use a default date
                    if null_percentage < 5:
                        # For small amounts, use median date
                        median_date = df[col].median()
                        df[col].fillna(median_date, inplace=True)
                        self.cleaning_log.append(f"Imputed {null_count} nulls in {col} with median date")
                    else:
                        self.cleaning_log.append(f"Left {null_count} nulls in datetime column {col} (standard practice)")
                
                else:
                    # Object/string columns
                    if null_percentage < 10:
                        df[col].fillna('Unknown', inplace=True)
                        self.cleaning_log.append(f"Imputed {null_count} nulls in {col} with 'Unknown'")
                    else:
                        self.cleaning_log.append(f"High null percentage in text column {col} ({null_percentage:.1f}%)")
            
            except Exception as e:
                self.cleaning_log.append(f"Null imputation failed for {col}: {e}")
        
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame, sheet_name: str, aggressive: bool = False) -> pd.DataFrame:
        """Detect and handle duplicate rows."""
        
        initial_count = len(df)
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count == 0:
            self.cleaning_log.append("No duplicate rows found")
            return df
        
        if aggressive or duplicate_count / len(df) > 0.05:  # More than 5% duplicates
            # Remove all duplicates
            df_cleaned = df.drop_duplicates()
            removed_count = len(df) - len(df_cleaned)
            self.cleaning_log.append(f"Removed {removed_count} duplicate rows ({removed_count/initial_count*100:.1f}%)")
            return df_cleaned
        else:
            # Conservative approach - flag but don't remove
            self.cleaning_log.append(f"Found {duplicate_count} duplicate rows ({duplicate_count/initial_count*100:.1f}%) - flagged for manual review")
            return df
    
    def _outlier_detection_treatment(self, df: pd.DataFrame, sheet_name: str, aggressive: bool = False) -> pd.DataFrame:
        """Detect and treat outliers using Z-score method."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                # Calculate Z-scores
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_threshold = 2.5 if aggressive else 3.0
                outliers = z_scores > outlier_threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_percentage = (outlier_count / len(df)) * 100
                    
                    if aggressive and outlier_percentage < 5:
                        # Cap outliers at percentiles
                        lower_cap = df[col].quantile(0.01)
                        upper_cap = df[col].quantile(0.99)
                        
                        original_outliers = df[col][outliers].copy()
                        df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
                        
                        self.cleaning_log.append(f"Capped {outlier_count} outliers in {col} to percentile range")
                    else:
                        self.cleaning_log.append(f"Found {outlier_count} outliers in {col} ({outlier_percentage:.1f}%) - flagged for review")
                        
            except Exception as e:
                self.cleaning_log.append(f"Outlier detection failed for {col}: {e}")
        
        return df
    
    def _validate_id_columns(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Validate ID column uniqueness and suggest corrections."""
        
        potential_id_cols = []
        
        for col in df.columns:
            # Check if column might be an ID
            if any(term in col.lower() for term in ['id', 'key', 'number', 'code']):
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8:  # High uniqueness suggests ID column
                    potential_id_cols.append(col)
        
        for col in potential_id_cols:
            unique_count = df[col].nunique()
            total_count = len(df)
            
            if unique_count == total_count:
                self.cleaning_log.append(f"✅ ID column {col} has perfect uniqueness")
            elif unique_count / total_count > 0.95:
                duplicate_count = total_count - unique_count
                self.cleaning_log.append(f"⚠️ ID column {col} has {duplicate_count} duplicates ({(1-unique_count/total_count)*100:.1f}%)")
                
                # Option to add suffix to make unique
                if duplicate_count < 10:  # Only for small numbers of duplicates
                    df[col] = df[col].astype(str)
                    df[col] = df[col] + '_' + df.groupby(col).cumcount().astype(str)
                    df[col] = df[col].str.replace('_0$', '', regex=True)  # Remove _0 suffix from first occurrence
                    self.cleaning_log.append(f"Made {col} unique by adding suffixes")
            else:
                self.cleaning_log.append(f"❌ Column {col} appears to be ID but has low uniqueness ({unique_count/total_count*100:.1f}%)")
        
        return df
    
    def _categorical_cleaning(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Clean categorical columns and handle rare values."""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if df[col].nunique() > 50:  # Skip high-cardinality columns
                continue
                
            try:
                value_counts = df[col].value_counts()
                total_count = len(df)
                
                # Find rare values (< 1% of data)
                rare_threshold = max(1, total_count * 0.01)
                rare_values = value_counts[value_counts < rare_threshold].index.tolist()
                
                if len(rare_values) > 0:
                    # Group rare values into "Other" category
                    df[col] = df[col].replace(rare_values, 'Other')
                    self.cleaning_log.append(f"Grouped {len(rare_values)} rare values in {col} into 'Other' category")
                
                # Clean text values
                if df[col].dtype == 'object':
                    # Standardize case
                    original_values = df[col].unique()
                    df[col] = df[col].astype(str).str.strip().str.title()
                    
                    # Remove extra whitespace
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                    
                    new_unique_count = df[col].nunique()
                    if len(original_values) != new_unique_count:
                        self.cleaning_log.append(f"Standardized text values in {col}: {len(original_values)} → {new_unique_count} unique values")
                
            except Exception as e:
                self.cleaning_log.append(f"Categorical cleaning failed for {col}: {e}")
        
        return df
    
    def _text_column_processing(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Process free text and high-uniqueness columns."""
        
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            uniqueness_ratio = df[col].nunique() / len(df)
            
            if uniqueness_ratio > 0.8:  # High uniqueness suggests free text
                try:
                    # Check average text length
                    avg_length = df[col].astype(str).str.len().mean()
                    
                    if avg_length > 50:  # Likely free text
                        # Basic text cleaning
                        df[col] = df[col].astype(str).str.strip()
                        
                        # Remove extra whitespace
                        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                        
                        # Flag for potential text analysis
                        self.cleaning_log.append(f"Cleaned free text column {col} (avg length: {avg_length:.1f} chars)")
                    else:
                        # Might be codes or IDs
                        self.cleaning_log.append(f"High-uniqueness column {col} identified (possible codes/IDs)")
                
                except Exception as e:
                    self.cleaning_log.append(f"Text processing failed for {col}: {e}")
        
        return df
    
    def _replace_placeholders(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Replace common placeholder values with NaN."""
        
        # Common placeholder values
        placeholders = [
            'n/a', 'na', 'null', 'none', 'undefined', 'missing', 'unknown',
            '', ' ', '-', '--', '---', 'tbd', 'tba', 'pending', 'void',
            'nil', 'blank', 'empty', '0000-00-00', '1900-01-01'
        ]
        
        placeholder_counts = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert to string and check for placeholders
                original_nulls = df[col].isnull().sum()
                
                # Case-insensitive replacement
                mask = df[col].astype(str).str.lower().isin([p.lower() for p in placeholders])
                placeholder_count = mask.sum()
                
                if placeholder_count > 0:
                    df.loc[mask, col] = np.nan
                    placeholder_counts[col] = placeholder_count
                    self.cleaning_log.append(f"Replaced {placeholder_count} placeholder values in {col} with NaN")
        
        if placeholder_counts:
            total_replaced = sum(placeholder_counts.values())
            self.cleaning_log.append(f"Total placeholder values replaced: {total_replaced}")
        
        return df
    
    def _ai_driven_cleanup(self, df: pd.DataFrame, original_df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Final AI-driven cleanup based on detected patterns."""
        
        if not self.client:
            return df
        
        try:
            # Analyze remaining data quality issues
            issues = []
            
            # Check for remaining high null percentages
            for col in df.columns:
                null_pct = (df[col].isnull().sum() / len(df)) * 100
                if null_pct > 30:
                    issues.append(f"{col}: {null_pct:.1f}% missing data")
            
            # Check for suspicious value patterns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() == 1:
                    issues.append(f"{col}: Only one unique value (consider removal)")
            
            # Check for columns with extreme outliers
            for col in df.select_dtypes(include=[np.number]).columns:
                if len(df[col].dropna()) > 0:
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
                    if outliers > len(df) * 0.1:  # More than 10% outliers
                        issues.append(f"{col}: {outliers} extreme outliers detected")
            
            if issues:
                context = f"""
Data Quality Issues Detected in {sheet_name}:

{chr(10).join(f"- {issue}" for issue in issues[:10])}

Dataset Info:
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Cleaning operations applied: {len(self.cleaning_log)}

Suggest final cleanup actions. Consider:
1. Column removal recommendations
2. Additional data transformations
3. Data validation rules
4. Business logic corrections

Return specific recommendations as JSON:
{{"actions": [{{"type": "remove_column", "column": "col_name", "reason": "explanation"}}]}}
"""
                
                messages = [
                    {"role": "system", "content": "You are a data quality expert who provides final cleanup recommendations for business datasets."},
                    {"role": "user", "content": context}
                ]
                
                response = chat_completion(self.client, messages, model="gpt-4o-mini")
                
                try:
                    recommendations = json.loads(response.strip())
                    actions = recommendations.get("actions", [])
                    
                    for action in actions:
                        action_type = action.get("type")
                        column = action.get("column")
                        reason = action.get("reason", "AI recommendation")
                        
                        if action_type == "remove_column" and column in df.columns:
                            # Only remove if very high missing data or single value
                            null_pct = (df[column].isnull().sum() / len(df)) * 100
                            if null_pct > 80 or df[column].nunique() == 1:
                                df = df.drop(columns=[column])
                                self.cleaning_log.append(f"AI recommendation: Removed column {column} - {reason}")
                        
                        # Add more action types as needed
                        
                except json.JSONDecodeError:
                    self.cleaning_log.append("AI cleanup recommendations could not be parsed")
                    
        except Exception as e:
            self.cleaning_log.append(f"AI-driven cleanup failed: {e}")
        
        return df
    
    def generate_cleaning_report(self, 
                               df_before: pd.DataFrame, 
                               df_after: pd.DataFrame, 
                               sheet_name: str) -> str:
        """Generate comprehensive cleaning report."""
        
        report_lines = [
            f"# Data Cleaning Report: {sheet_name}",
            f"",
            f"**Operations Performed**: {len(self.cleaning_log)}",
            f"**Original Size**: {len(df_before):,} rows × {len(df_before.columns)} columns",
            f"**Final Size**: {len(df_after):,} rows × {len(df_after.columns)} columns",
            f"",
            "## Impact Summary",
            f"",
            f"- **Rows Changed**: {len(df_before) - len(df_after):+,}",
            f"- **Columns Changed**: {len(df_before.columns) - len(df_after.columns):+,}",
        ]
        
        # Missing data improvement
        missing_before = df_before.isnull().sum().sum()
        missing_after = df_after.isnull().sum().sum()
        missing_improvement = missing_before - missing_after
        
        report_lines.extend([
            f"- **Missing Values Reduced**: {missing_improvement:,}",
            f"- **Data Completeness**: {((len(df_before)*len(df_before.columns) - missing_before)/(len(df_before)*len(df_before.columns))*100):.1f}% → {((len(df_after)*len(df_after.columns) - missing_after)/(len(df_after)*len(df_after.columns))*100):.1f}%",
            f"",
            "## Operations Log",
            f""
        ])
        
        for i, operation in enumerate(self.cleaning_log, 1):
            report_lines.append(f"{i}. {operation}")
        
        return "\n".join(report_lines)


# Convenience function
def run_smart_autofix(df: pd.DataFrame, 
                     sheet_name: str = "Sheet1",
                     aggressive_mode: bool = False) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Convenience function to run smart auto-fix system.
    
    Returns:
        Tuple of (cleaned_DataFrame, operations_log, cleaning_report)
    """
    
    autofix_system = SmartAutoFixSystem()
    df_before = df.copy()
    
    df_cleaned, operations_log = autofix_system.run_comprehensive_autofix(
        df, sheet_name, aggressive_mode
    )
    
    cleaning_report = autofix_system.generate_cleaning_report(
        df_before, df_cleaned, sheet_name
    )
    
    return df_cleaned, operations_log, cleaning_report
