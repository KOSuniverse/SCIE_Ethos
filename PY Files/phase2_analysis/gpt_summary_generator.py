# gpt_summary_generator.py
"""
AI-Powered Summary Generator matching the original Colab workflow.
Provides business-focused summaries with financial insights and recommendations.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
import time

# LLM integration
try:
    from llm_client import get_openai_client, chat_completion
except ImportError:
    print("⚠️ LLM modules not available - summaries will be basic")
    get_openai_client = None
    chat_completion = None

class GPTSummaryGenerator:
    """
    AI-powered summary generator for business stakeholders.
    Creates executive summaries, data quality reports, and actionable insights.
    """
    
    def __init__(self):
        self.client = None
        if get_openai_client:
            try:
                self.client = get_openai_client()
            except Exception as e:
                print(f"⚠️ Could not initialize LLM client: {e}")
    
    def generate_executive_summary(self, 
                                  df: pd.DataFrame, 
                                  sheet_name: str,
                                  filename: str,
                                  metadata: Dict = None,
                                  eda_results: Dict = None) -> str:
        """
        Generate comprehensive executive summary matching Colab workflow.
        """
        
        if not self.client:
            return self._generate_basic_executive_summary(df, sheet_name, filename, metadata, eda_results)
        
        try:
            # Prepare context for AI
            context = self._build_summary_context(df, sheet_name, filename, metadata, eda_results)
            
            prompt = f"""
You are a senior business analyst creating an executive summary for supply chain stakeholders.

{context}

Create a comprehensive executive summary that includes:

1. **Data Overview**: Dataset size, scope, and quality assessment
2. **Key Financial Metrics**: Total values, cost centers, and financial highlights
3. **Supply Chain Insights**: Inventory, WIP, costs, and operational metrics
4. **Data Quality Assessment**: Completeness, accuracy, and reliability
5. **Business Recommendations**: Actionable next steps and areas for investigation
6. **Risk Indicators**: Data quality issues, outliers, or anomalies requiring attention

Write this for C-level executives who need clear, actionable insights. Use specific numbers and percentages.
Keep the tone professional and focus on business impact.
"""
            
            messages = [
                {"role": "system", "content": "You are an expert supply chain business analyst with 15+ years of experience in data analysis and executive reporting."},
                {"role": "user", "content": prompt}
            ]
            
            summary = chat_completion(self.client, messages, model="gpt-4o")
            return summary.strip()
            
        except Exception as e:
            print(f"⚠️ AI executive summary failed: {e}")
            return self._generate_basic_executive_summary(df, sheet_name, filename, metadata, eda_results)
    
    def generate_data_quality_report(self, 
                                   df: pd.DataFrame, 
                                   sheet_name: str,
                                   cleaning_log: List[str] = None) -> str:
        """
        Generate detailed data quality report with AI insights.
        """
        
        if not self.client:
            return self._generate_basic_quality_report(df, sheet_name, cleaning_log)
        
        try:
            # Analyze data quality
            quality_metrics = self._analyze_data_quality(df)
            
            context = f"""
Data Quality Analysis for: {sheet_name}

Dataset Size: {len(df):,} rows × {len(df.columns)} columns
Missing Data: {quality_metrics['missing_percentage']:.1f}%
Duplicate Rows: {quality_metrics['duplicate_count']} ({quality_metrics['duplicate_percentage']:.1f}%)
Completeness Score: {quality_metrics['completeness_score']:.1f}%

Column Analysis:
{self._format_column_quality(df, quality_metrics)}

{f"Cleaning Operations Performed: {len(cleaning_log)}" if cleaning_log else "No cleaning operations logged"}
"""
            
            prompt = f"""
Analyze this data quality assessment and provide a business-focused report.

{context}

Create a data quality report that includes:

1. **Overall Quality Score**: Grade the dataset (A-F) with justification
2. **Critical Issues**: Data problems that could impact business decisions
3. **Data Reliability**: Assessment of data trustworthiness for analysis
4. **Recommended Actions**: Specific steps to improve data quality
5. **Impact Assessment**: How data quality issues affect business insights

Focus on practical implications for business users. Be specific about what actions to take.
"""
            
            messages = [
                {"role": "system", "content": "You are a data quality expert with extensive experience in supply chain data analysis."},
                {"role": "user", "content": prompt}
            ]
            
            report = chat_completion(self.client, messages, model="gpt-4o-mini")
            return report.strip()
            
        except Exception as e:
            print(f"⚠️ AI quality report failed: {e}")
            return self._generate_basic_quality_report(df, sheet_name, cleaning_log)
    
    def generate_eda_insights_summary(self, 
                                    df: pd.DataFrame,
                                    eda_results: Dict,
                                    sheet_name: str) -> str:
        """
        Generate AI-powered insights from EDA results.
        """
        
        if not self.client:
            return self._generate_basic_eda_summary(df, eda_results, sheet_name)
        
        try:
            # Extract key findings from EDA
            insights = []
            charts_generated = len(eda_results.get("chart_paths", []))
            rounds_completed = len(eda_results.get("rounds", []))
            
            for round_data in eda_results.get("rounds", []):
                round_insights = round_data.get("insights", [])
                insights.extend(round_insights)
            
            business_insights = eda_results.get("business_insights", {})
            supply_chain_metrics = business_insights.get("supply_chain_metrics", {})
            
            context = f"""
EDA Analysis Results for: {sheet_name}

Analysis Rounds Completed: {rounds_completed}
Charts Generated: {charts_generated}

Supply Chain Metrics Found:
{json.dumps(supply_chain_metrics, indent=2) if supply_chain_metrics else "None identified"}

Key Insights from Analysis:
{chr(10).join(f"- {insight}" for insight in insights[:10])}

Data Characteristics:
- Records: {len(df):,}
- Numeric Columns: {len(df.select_dtypes(include=['number']).columns)}
- Categorical Columns: {len(df.select_dtypes(include=['object', 'category']).columns)}
"""
            
            prompt = f"""
Summarize these EDA findings for business stakeholders.

{context}

Create an insights summary that includes:

1. **Key Discoveries**: Most important patterns and relationships found
2. **Supply Chain Implications**: What these findings mean for operations
3. **Financial Insights**: Cost, value, and financial metric highlights
4. **Operational Patterns**: Trends, outliers, and operational insights
5. **Strategic Recommendations**: Actions to take based on the analysis

Write for business users who need to understand what the data reveals about their operations.
Focus on actionable insights that can drive business decisions.
"""
            
            messages = [
                {"role": "system", "content": "You are a supply chain consultant specializing in data-driven insights and operational improvements."},
                {"role": "user", "content": prompt}
            ]
            
            summary = chat_completion(self.client, messages, model="gpt-4o-mini")
            return summary.strip()
            
        except Exception as e:
            print(f"⚠️ AI EDA insights failed: {e}")
            return self._generate_basic_eda_summary(df, eda_results, sheet_name)
    
    def generate_autofix_summary(self, 
                                cleaning_operations: List[str],
                                df_before: pd.DataFrame,
                                df_after: pd.DataFrame,
                                sheet_name: str) -> str:
        """
        Generate summary of auto-fix operations performed.
        """
        
        if not self.client:
            return self._generate_basic_autofix_summary(cleaning_operations, df_before, df_after, sheet_name)
        
        try:
            # Calculate impact metrics
            rows_before = len(df_before)
            rows_after = len(df_after)
            cols_before = len(df_before.columns)
            cols_after = len(df_after.columns)
            
            missing_before = df_before.isnull().sum().sum()
            missing_after = df_after.isnull().sum().sum()
            
            context = f"""
Auto-Fix Operations Summary for: {sheet_name}

Operations Performed: {len(cleaning_operations)}
{chr(10).join(f"- {op}" for op in cleaning_operations[:10])}

Impact Metrics:
- Rows: {rows_before:,} → {rows_after:,} (Change: {rows_after - rows_before:+,})
- Columns: {cols_before} → {cols_after} (Change: {cols_after - cols_before:+})
- Missing Values: {missing_before:,} → {missing_after:,} (Reduction: {missing_before - missing_after:,})
- Missing Percentage: {(missing_before/(rows_before*cols_before)*100):.1f}% → {(missing_after/(rows_after*cols_after)*100):.1f}%
"""
            
            prompt = f"""
Summarize the data cleaning and auto-fix operations for business stakeholders.

{context}

Create a summary that includes:

1. **Operations Overview**: What cleaning steps were performed and why
2. **Quality Improvements**: How the data quality improved
3. **Impact Assessment**: What the changes mean for data reliability
4. **Business Benefits**: How cleaner data improves analysis accuracy
5. **Recommendations**: Additional cleaning steps that might be needed

Focus on the business value of the data cleaning process. Explain why these operations matter for decision-making.
"""
            
            messages = [
                {"role": "system", "content": "You are a data engineering expert who specializes in explaining technical processes to business stakeholders."},
                {"role": "user", "content": prompt}
            ]
            
            summary = chat_completion(self.client, messages, model="gpt-4o-mini")
            return summary.strip()
            
        except Exception as e:
            print(f"⚠️ AI autofix summary failed: {e}")
            return self._generate_basic_autofix_summary(cleaning_operations, df_before, df_after, sheet_name)
    
    def _build_summary_context(self, 
                              df: pd.DataFrame, 
                              sheet_name: str,
                              filename: str,
                              metadata: Dict = None,
                              eda_results: Dict = None) -> str:
        """Build comprehensive context for AI summary generation."""
        
        # Basic dataset info
        context_lines = [
            f"Dataset: {filename} - {sheet_name}",
            f"Size: {len(df):,} rows × {len(df.columns)} columns",
            f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        # Data quality metrics
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        duplicate_count = df.duplicated().sum()
        
        context_lines.extend([
            f"Missing Data: {missing_pct:.1f}%",
            f"Duplicate Rows: {duplicate_count} ({duplicate_count/len(df)*100:.1f}%)"
        ])
        
        # Column analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        context_lines.extend([
            f"Numeric Columns: {len(numeric_cols)}",
            f"Categorical Columns: {len(categorical_cols)}"
        ])
        
        # Supply chain metrics
        supply_cols = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['wip', 'work in progress']):
                supply_cols.setdefault('WIP', []).append(col)
            elif any(term in col_lower for term in ['cost', 'value', 'price']):
                supply_cols.setdefault('Financial', []).append(col)
            elif any(term in col_lower for term in ['qty', 'quantity', 'inventory']):
                supply_cols.setdefault('Inventory', []).append(col)
        
        if supply_cols:
            context_lines.append("Supply Chain Columns Identified:")
            for category, cols in supply_cols.items():
                total_value = 0
                numeric_supply_cols = [col for col in cols if col in numeric_cols]
                if numeric_supply_cols:
                    try:
                        total_value = float(df[numeric_supply_cols].sum().sum())
                    except:
                        pass
                context_lines.append(f"- {category}: {len(cols)} columns, Total Value: ${total_value:,.2f}")
        
        # EDA results if available
        if eda_results:
            chart_count = len(eda_results.get("chart_paths", []))
            rounds_count = len(eda_results.get("rounds", []))
            context_lines.extend([
                f"EDA Analysis: {rounds_count} rounds completed",
                f"Charts Generated: {chart_count}"
            ])
        
        # Metadata if available
        if metadata:
            quality_flags = metadata.get("quality_flags", [])
            if quality_flags:
                context_lines.append(f"Quality Flags: {', '.join(quality_flags)}")
        
        return "\n".join(context_lines)
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze data quality metrics."""
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        return {
            "missing_percentage": (missing_cells / total_cells) * 100,
            "duplicate_count": int(duplicate_rows),
            "duplicate_percentage": (duplicate_rows / len(df)) * 100,
            "completeness_score": ((total_cells - missing_cells) / total_cells) * 100
        }
    
    def _format_column_quality(self, df: pd.DataFrame, quality_metrics: Dict) -> str:
        """Format column quality information."""
        
        lines = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            quality_score = "Good" if null_pct < 5 else "Fair" if null_pct < 20 else "Poor"
            lines.append(f"- {col}: {null_pct:.1f}% missing, {unique_count} unique values ({quality_score})")
        
        return "\n".join(lines[:10])  # Limit to top 10 columns
    
    def _generate_basic_executive_summary(self, 
                                        df: pd.DataFrame, 
                                        sheet_name: str,
                                        filename: str,
                                        metadata: Dict = None,
                                        eda_results: Dict = None) -> str:
        """Generate basic executive summary when AI is not available."""
        
        lines = [
            f"# Executive Summary: {sheet_name}",
            "",
            f"**Source**: {filename}",
            f"**Records**: {len(df):,} rows",
            f"**Columns**: {len(df.columns)}",
            f"**Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Data Overview",
            ""
        ]
        
        # Data quality
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        duplicate_count = df.duplicated().sum()
        
        lines.extend([
            f"- **Data Completeness**: {100 - missing_pct:.1f}%",
            f"- **Missing Data**: {missing_pct:.1f}%",
            f"- **Duplicate Records**: {duplicate_count} ({duplicate_count/len(df)*100:.1f}%)",
            ""
        ])
        
        # Column analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        lines.extend([
            "## Column Analysis",
            "",
            f"- **Numeric Columns**: {len(numeric_cols)}",
            f"- **Categorical Columns**: {len(categorical_cols)}",
            ""
        ])
        
        # Supply chain metrics
        supply_metrics = self._extract_basic_supply_metrics(df)
        if supply_metrics:
            lines.extend([
                "## Supply Chain Metrics",
                ""
            ])
            for category, info in supply_metrics.items():
                lines.append(f"- **{category}**: ${info['total_value']:,.2f} across {info['column_count']} columns")
            lines.append("")
        
        # EDA summary if available
        if eda_results:
            chart_count = len(eda_results.get("chart_paths", []))
            rounds_count = len(eda_results.get("rounds", []))
            lines.extend([
                "## Analysis Results",
                "",
                f"- **Analysis Rounds**: {rounds_count}",
                f"- **Charts Generated**: {chart_count}",
                ""
            ])
        
        lines.extend([
            "## Recommendations",
            "",
            "- Review data quality issues identified above",
            "- Investigate duplicate records if percentage is high",
            "- Consider additional data validation for critical columns",
            "- Proceed with detailed analysis on clean dataset"
        ])
        
        return "\n".join(lines)
    
    def _generate_basic_quality_report(self, 
                                     df: pd.DataFrame, 
                                     sheet_name: str,
                                     cleaning_log: List[str] = None) -> str:
        """Generate basic quality report when AI is not available."""
        
        quality_metrics = self._analyze_data_quality(df)
        
        # Determine quality grade
        completeness = quality_metrics["completeness_score"]
        grade = "A" if completeness >= 95 else "B" if completeness >= 85 else "C" if completeness >= 70 else "D" if completeness >= 50 else "F"
        
        lines = [
            f"# Data Quality Report: {sheet_name}",
            "",
            f"**Overall Quality Grade**: {grade}",
            f"**Completeness Score**: {completeness:.1f}%",
            "",
            "## Quality Metrics",
            "",
            f"- **Missing Data**: {quality_metrics['missing_percentage']:.1f}%",
            f"- **Duplicate Rows**: {quality_metrics['duplicate_count']} ({quality_metrics['duplicate_percentage']:.1f}%)",
            ""
        ]
        
        # Column quality breakdown
        lines.extend([
            "## Column Quality Assessment",
            ""
        ])
        
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            quality_status = "✅ Good" if null_pct < 5 else "⚠️ Fair" if null_pct < 20 else "❌ Poor"
            lines.append(f"- **{col}**: {null_pct:.1f}% missing ({quality_status})")
        
        lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        if quality_metrics["missing_percentage"] > 10:
            lines.append("- High missing data rate requires investigation")
        if quality_metrics["duplicate_percentage"] > 1:
            lines.append("- Review duplicate records for data entry issues")
        if completeness < 85:
            lines.append("- Consider data source improvements")
        
        return "\n".join(lines)
    
    def _generate_basic_eda_summary(self, 
                                  df: pd.DataFrame,
                                  eda_results: Dict,
                                  sheet_name: str) -> str:
        """Generate basic EDA summary when AI is not available."""
        
        lines = [
            f"# EDA Insights Summary: {sheet_name}",
            "",
            f"**Dataset Size**: {len(df):,} rows × {len(df.columns)} columns",
            ""
        ]
        
        # Analysis overview
        charts_generated = len(eda_results.get("chart_paths", []))
        rounds_completed = len(eda_results.get("rounds", []))
        
        lines.extend([
            "## Analysis Overview",
            "",
            f"- **Rounds Completed**: {rounds_completed}",
            f"- **Charts Generated**: {charts_generated}",
            ""
        ])
        
        # Extract insights
        all_insights = []
        for round_data in eda_results.get("rounds", []):
            round_insights = round_data.get("insights", [])
            all_insights.extend(round_insights)
        
        if all_insights:
            lines.extend([
                "## Key Findings",
                ""
            ])
            for insight in all_insights[:5]:
                lines.append(f"- {insight}")
            lines.append("")
        
        # Supply chain metrics
        business_insights = eda_results.get("business_insights", {})
        supply_chain_metrics = business_insights.get("supply_chain_metrics", {})
        
        if supply_chain_metrics:
            lines.extend([
                "## Supply Chain Insights",
                ""
            ])
            for category, metrics in supply_chain_metrics.items():
                if metrics.get("total_value", 0) > 0:
                    lines.append(f"- **{category.title()}**: ${metrics['total_value']:,.2f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_basic_autofix_summary(self, 
                                      cleaning_operations: List[str],
                                      df_before: pd.DataFrame,
                                      df_after: pd.DataFrame,
                                      sheet_name: str) -> str:
        """Generate basic autofix summary when AI is not available."""
        
        lines = [
            f"# Auto-Fix Operations Summary: {sheet_name}",
            "",
            f"**Operations Performed**: {len(cleaning_operations)}",
            ""
        ]
        
        # Impact metrics
        rows_change = len(df_after) - len(df_before)
        cols_change = len(df_after.columns) - len(df_before.columns)
        missing_before = df_before.isnull().sum().sum()
        missing_after = df_after.isnull().sum().sum()
        missing_reduction = missing_before - missing_after
        
        lines.extend([
            "## Impact Summary",
            "",
            f"- **Rows**: {len(df_before):,} → {len(df_after):,} ({rows_change:+,})",
            f"- **Columns**: {len(df_before.columns)} → {len(df_after.columns)} ({cols_change:+})",
            f"- **Missing Values**: {missing_before:,} → {missing_after:,} (-{missing_reduction:,})",
            ""
        ])
        
        # Operations performed
        if cleaning_operations:
            lines.extend([
                "## Operations Performed",
                ""
            ])
            for op in cleaning_operations[:10]:
                lines.append(f"- {op}")
            lines.append("")
        
        lines.extend([
            "## Data Quality Improvement",
            "",
            f"- Missing data reduced by {missing_reduction:,} values",
            f"- Data completeness improved",
            f"- Dataset ready for analysis"
        ])
        
        return "\n".join(lines)
    
    def _extract_basic_supply_metrics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Extract basic supply chain metrics without AI."""
        
        metrics = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Define supply chain categories
        categories = {
            'WIP': ['wip', 'work in progress'],
            'Cost/Value': ['cost', 'value', 'price', 'amount'],
            'Inventory': ['inventory', 'stock', 'qty', 'quantity'],
        }
        
        for category, keywords in categories.items():
            matching_cols = []
            for col in numeric_cols:
                if any(keyword in str(col).lower() for keyword in keywords):
                    matching_cols.append(col)
            
            if matching_cols:
                try:
                    total_value = float(df[matching_cols].sum().sum())
                    metrics[category] = {
                        'total_value': total_value,
                        'column_count': len(matching_cols),
                        'columns': matching_cols
                    }
                except:
                    pass
        
        return metrics


# Convenience function
def generate_comprehensive_summary(df: pd.DataFrame,
                                 sheet_name: str,
                                 filename: str,
                                 metadata: Dict = None,
                                 eda_results: Dict = None,
                                 cleaning_log: List[str] = None) -> Dict[str, str]:
    """
    Generate comprehensive summaries matching the original Colab workflow.
    
    Returns:
        Dictionary containing different types of summaries
    """
    
    generator = GPTSummaryGenerator()
    
    summaries = {
        "executive_summary": generator.generate_executive_summary(
            df, sheet_name, filename, metadata, eda_results
        ),
        "data_quality_report": generator.generate_data_quality_report(
            df, sheet_name, cleaning_log
        )
    }
    
    if eda_results:
        summaries["eda_insights"] = generator.generate_eda_insights_summary(
            df, eda_results, sheet_name
        )
    
    if cleaning_log:
        # For autofix summary, we'd need before/after DataFrames
        # This is a placeholder for when that data is available
        summaries["autofix_summary"] = f"Performed {len(cleaning_log)} cleaning operations"
    
    return summaries
