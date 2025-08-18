# enhanced_eda_system.py
"""
Comprehensive EDA System matching the original Colab workflow.
This module provides AI-powered, multi-round EDA with detailed business summaries.
"""

import json
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# LLM integration
try:
    from llm_client import get_openai_client, chat_completion
    from llm_prompts import build_scaffold_messages, minimal_tools_catalog
except ImportError:
    print("⚠️ LLM modules not available - some features will be limited")
    get_openai_client = None
    chat_completion = None

# Cloud utilities
try:
    from dbx_utils import upload_bytes, upload_json
except ImportError:
    print("⚠️ Cloud utilities not available - charts will save locally")
    upload_bytes = None
    upload_json = None

# Profiling tools
try:
    import ydata_profiling
    PROFILING_AVAILABLE = True
except ImportError:
    print("⚠️ YData Profiling not available - install with: pip install ydata-profiling")
    PROFILING_AVAILABLE = False

class EnhancedEDASystem:
    """
    Comprehensive EDA system with AI-powered analysis and business insights.
    Matches the original Colab workflow with multi-round analysis.
    """
    
    def __init__(self, charts_folder: str = None, summaries_folder: str = None):
        self.charts_folder = charts_folder or "/Project_Root/04_Data/02_EDA_Charts"
        self.summaries_folder = summaries_folder or "/Project_Root/04_Data/03_Summaries"
        self.client = None
        self.round_counter = 1
        
        # Initialize LLM client if available
        if get_openai_client:
            try:
                self.client = get_openai_client()
            except Exception as e:
                print(f"⚠️ Could not initialize LLM client: {e}")
    
    def run_comprehensive_eda(self, 
                            df: pd.DataFrame, 
                            sheet_name: str = "Sheet1",
                            filename: str = "data.xlsx",
                            max_rounds: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive multi-round EDA analysis matching the original Colab workflow.
        
        Returns:
            Dict with all analysis results, summaries, and chart paths
        """
        
        print(f"🧠 Starting Comprehensive EDA for {sheet_name} ({len(df)} rows, {len(df.columns)} columns)")
        
        # Initialize results container
        results = {
            "metadata": self._generate_metadata_summary(df, sheet_name, filename),
            "data_profiling": {},
            "rounds": [],
            "final_summary": {},
            "chart_paths": [],
            "business_insights": {}
        }
        
        # STEP 1: Data Profiling & Quality Assessment
        print("📊 STEP 1: Data Profiling & Quality Assessment")
        results["data_profiling"] = self._comprehensive_data_profiling(df, sheet_name)
        
        # STEP 2: Generate YData Profiling Report (if available)
        if PROFILING_AVAILABLE:
            print("📋 STEP 2: Generating YData Profiling Report")
            results["profiling_report_path"] = self._generate_ydata_profile(df, sheet_name)
        
        # STEP 3: Initial EDA Round with Basic Charts
        print("🎨 STEP 3: Initial EDA Round - Basic Analysis")
        round_1_results = self._execute_eda_round(df, sheet_name, round_num=1, 
                                                 analysis_type="basic_exploration")
        results["rounds"].append(round_1_results)
        results["chart_paths"].extend(round_1_results.get("chart_paths", []))
        
        # STEP 4: AI-Powered Follow-up Analysis (if LLM available)
        if self.client and max_rounds > 1:
            for round_num in range(2, max_rounds + 1):
                print(f"🧠 STEP {round_num + 2}: AI-Powered Follow-up Round {round_num}")
                
                # Get AI suggestions based on previous rounds
                follow_up_results = self._ai_driven_followup(df, sheet_name, 
                                                           results["rounds"], round_num)
                results["rounds"].append(follow_up_results)
                results["chart_paths"].extend(follow_up_results.get("chart_paths", []))
        
        # STEP 5: Business Insights & Supply Chain Analysis
        print("💼 STEP 5: Business Insights & Supply Chain Analysis")
        results["business_insights"] = self._generate_business_insights(df, sheet_name, results)
        
        # STEP 6: Final AI Summary (if available)
        if self.client:
            print("📝 STEP 6: Generating AI Executive Summary")
            results["final_summary"] = self._generate_ai_executive_summary(df, sheet_name, results)
        else:
            results["final_summary"] = self._generate_basic_summary(df, sheet_name, results)
        
        # STEP 7: Save comprehensive results
        print("💾 STEP 7: Saving Comprehensive Results")
        self._save_comprehensive_results(results, sheet_name, filename)
        
        print(f"✅ Comprehensive EDA Complete! Generated {len(results['chart_paths'])} charts")
        return results
    
    def _generate_metadata_summary(self, df: pd.DataFrame, sheet_name: str, filename: str) -> Dict[str, Any]:
        """Generate comprehensive metadata summary"""
        
        # Basic stats
        metadata = {
            "filename": filename,
            "sheet_name": sheet_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "column_types": df.dtypes.astype(str).to_dict(),
            "missing_data": df.isnull().sum().to_dict(),
            "missing_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
        }
        
        # Column analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        metadata.update({
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "high_cardinality_cols": [col for col in categorical_cols if df[col].nunique() > 50],
            "potential_ids": [col for col in df.columns if df[col].nunique() == len(df) and len(df) > 1]
        })
        
        # Data quality flags
        quality_flags = []
        if metadata["missing_percentage"] > 20:
            quality_flags.append("HIGH_MISSING_DATA")
        if len(metadata["potential_ids"]) > 0:
            quality_flags.append("POTENTIAL_ID_COLUMNS")
        if any(df.duplicated()):
            quality_flags.append("DUPLICATE_ROWS")
        
        metadata["quality_flags"] = quality_flags
        
        return metadata
    
    def _comprehensive_data_profiling(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Comprehensive data profiling similar to original Colab workflow"""
        
        profiling = {
            "basic_stats": {},
            "column_profiles": {},
            "correlations": {},
            "outliers": {},
            "data_quality": {}
        }
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            profiling["basic_stats"] = df[numeric_cols].describe().to_dict()
        
        # Column-by-column profiling
        for col in df.columns:
            col_profile = {
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2)
            }
            
            if df[col].dtype in [np.number]:
                # Numeric column analysis
                col_profile.update({
                    "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                    "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
                    "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                    "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                    "skewness": float(df[col].skew()) if pd.notna(df[col].skew()) else None,
                    "kurtosis": float(df[col].kurtosis()) if pd.notna(df[col].kurtosis()) else None
                })
                
                # Outlier detection using IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                col_profile["outlier_count"] = int(outlier_mask.sum())
                col_profile["outlier_percentage"] = round((outlier_mask.sum() / len(df)) * 100, 2)
                
            elif df[col].dtype == 'object':
                # Categorical column analysis
                value_counts = df[col].value_counts()
                col_profile.update({
                    "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 and len(value_counts) > 0 else 0,
                    "cardinality": int(df[col].nunique()),
                    "cardinality_ratio": round(df[col].nunique() / len(df), 4)
                })
            
            profiling["column_profiles"][col] = col_profile
        
        # Correlation analysis for numeric columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr()
                # Find strong correlations (> 0.7 or < -0.7)
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_correlations.append({
                                "column1": corr_matrix.columns[i],
                                "column2": corr_matrix.columns[j],
                                "correlation": round(float(corr_val), 3)
                            })
                
                profiling["correlations"] = {
                    "strong_correlations": strong_correlations,
                    "correlation_matrix": corr_matrix.round(3).to_dict()
                }
            except Exception as e:
                profiling["correlations"] = {"error": str(e)}
        
        return profiling
    
    def _generate_ydata_profile(self, df: pd.DataFrame, sheet_name: str) -> Optional[str]:
        """Generate YData Profiling HTML report if available"""
        
        if not PROFILING_AVAILABLE:
            return None
        
        try:
            # Create profiling report
            profile = ydata_profiling.ProfileReport(
                df, 
                title=f"Data Profile: {sheet_name}",
                explorative=True,
                dark_mode=False
            )
            
            # Generate HTML
            html_content = profile.to_html()
            
            # Save to cloud if possible
            report_path = f"{self.summaries_folder}/profile_{sheet_name.replace(' ', '_')}.html"
            
            if upload_bytes:
                upload_bytes(report_path, html_content.encode('utf-8'))
                print(f"📋 YData Profile saved: {report_path}")
                return report_path
            else:
                # Save locally as fallback
                local_path = f"profile_{sheet_name.replace(' ', '_')}.html"
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"📋 YData Profile saved locally: {local_path}")
                return local_path
                
        except Exception as e:
            print(f"⚠️ YData Profiling failed: {e}")
            return None
    
    def _execute_eda_round(self, 
                          df: pd.DataFrame, 
                          sheet_name: str, 
                          round_num: int,
                          analysis_type: str = "basic_exploration",
                          suggested_actions: List[Dict] = None) -> Dict[str, Any]:
        """Execute a round of EDA analysis with chart generation"""
        
        round_results = {
            "round_number": round_num,
            "analysis_type": analysis_type,
            "chart_paths": [],
            "insights": [],
            "ai_summary": None
        }
        
        # Determine actions for this round
        if suggested_actions:
            actions = suggested_actions
        else:
            actions = self._generate_default_actions(df, analysis_type)
        
        print(f"  📊 Executing {len(actions)} analysis actions...")
        
        # Execute each action
        for i, action in enumerate(actions):
            try:
                chart_path = self._execute_single_action(df, action, sheet_name, round_num, i)
                if chart_path:
                    round_results["chart_paths"].append(chart_path)
                    
                # Generate insight for this action
                insight = self._generate_action_insight(df, action)
                if insight:
                    round_results["insights"].append(insight)
                    
            except Exception as e:
                print(f"    ⚠️ Action {action.get('action', 'unknown')} failed: {e}")
        
        # Generate AI summary for this round if client available
        if self.client:
            round_results["ai_summary"] = self._generate_round_ai_summary(
                df, sheet_name, round_results, round_num
            )
        
        return round_results
    
    def _generate_default_actions(self, df: pd.DataFrame, analysis_type: str) -> List[Dict]:
        """Generate default EDA actions based on data characteristics"""
        
        actions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if analysis_type == "basic_exploration":
            # Basic exploration actions
            
            # Distribution analysis for top numeric columns
            for col in numeric_cols[:3]:
                actions.append({"action": "histogram", "column": col})
                actions.append({"action": "boxplot", "column": col})
            
            # Correlation analysis if enough numeric columns
            if len(numeric_cols) >= 2:
                actions.append({"action": "correlation_matrix"})
            
            # Scatter plots for numeric pairs
            if len(numeric_cols) >= 2:
                actions.append({
                    "action": "scatter", 
                    "x": numeric_cols[0], 
                    "y": numeric_cols[1]
                })
            
            # Top N analysis for categorical + numeric
            if categorical_cols and numeric_cols:
                actions.append({
                    "action": "groupby_topn",
                    "group": categorical_cols[0],
                    "metric": numeric_cols[0],
                    "topn": 10
                })
        
        elif analysis_type == "deep_dive":
            # Deep dive actions
            
            # Multiple scatter plots
            for i in range(min(3, len(numeric_cols) - 1)):
                for j in range(i + 1, min(i + 3, len(numeric_cols))):
                    actions.append({
                        "action": "scatter",
                        "x": numeric_cols[i],
                        "y": numeric_cols[j]
                    })
            
            # Groupby analyses for different combinations
            for cat_col in categorical_cols[:2]:
                for num_col in numeric_cols[:2]:
                    actions.append({
                        "action": "groupby_topn",
                        "group": cat_col,
                        "metric": num_col,
                        "topn": 15
                    })
        
        # Supply chain specific actions
        supply_keywords = ['wip', 'inventory', 'cost', 'value', 'qty', 'quantity']
        if any(keyword in ' '.join(df.columns).lower() for keyword in supply_keywords):
            actions.append({
                "action": "supply_chain_dashboard",
                "title": f"Supply Chain Analysis: {analysis_type}"
            })
        
        return actions
    
    def _execute_single_action(self, 
                              df: pd.DataFrame, 
                              action: Dict, 
                              sheet_name: str, 
                              round_num: int, 
                              action_idx: int) -> Optional[str]:
        """Execute a single EDA action and return chart path"""
        
        action_type = action.get("action")
        suffix = f"_{sheet_name.replace(' ', '_')}_r{round_num}_{action_idx}"
        
        try:
            if action_type == "histogram":
                return self._create_histogram(df, action, suffix)
            elif action_type == "boxplot":
                return self._create_boxplot(df, action, suffix)
            elif action_type == "scatter":
                return self._create_scatter(df, action, suffix)
            elif action_type == "correlation_matrix":
                return self._create_correlation_matrix(df, action, suffix)
            elif action_type == "groupby_topn":
                return self._create_groupby_chart(df, action, suffix)
            elif action_type == "supply_chain_dashboard":
                return self._create_supply_chain_dashboard(df, action, suffix)
            else:
                print(f"    ⚠️ Unknown action type: {action_type}")
                return None
                
        except Exception as e:
            print(f"    ⚠️ Failed to execute {action_type}: {e}")
            return None
    
    def _create_histogram(self, df: pd.DataFrame, action: Dict, suffix: str) -> Optional[str]:
        """Create enhanced histogram with business insights"""
        
        col = action.get("column")
        if not col or col not in df.columns:
            return None
        
        # Clean data
        clean_data = df[col].dropna()
        if len(clean_data) == 0:
            return None
        
        # Create figure with enhanced styling
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('default')
        
        # Create histogram with better binning
        n_bins = min(50, max(10, len(clean_data) // 20))
        counts, bins, patches = ax.hist(clean_data, bins=n_bins, alpha=0.7, 
                                       color='skyblue', edgecolor='black', linewidth=0.5)
        
        # Add statistics overlay
        mean_val = clean_data.mean()
        median_val = clean_data.median()
        std_val = clean_data.std()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Enhanced title and labels
        ax.set_title(f'Distribution Analysis: {col}\n'
                    f'μ={mean_val:.2f}, σ={std_val:.2f}, Skew={clean_data.skew():.2f}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save chart
        chart_path = f"{self.charts_folder}/histogram_{col.replace(' ', '_')}{suffix}.png"
        return self._save_chart(fig, chart_path)
    
    def _create_boxplot(self, df: pd.DataFrame, action: Dict, suffix: str) -> Optional[str]:
        """Create enhanced boxplot with outlier analysis"""
        
        col = action.get("column")
        if not col or col not in df.columns:
            return None
        
        clean_data = df[col].dropna()
        if len(clean_data) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create boxplot with enhanced styling
        box_plot = ax.boxplot(clean_data, patch_artist=True, 
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2),
                             flierprops=dict(marker='o', markersize=4, alpha=0.6))
        
        # Add statistics
        Q1, Q3 = clean_data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = clean_data[(clean_data < Q1 - 1.5*IQR) | (clean_data > Q3 + 1.5*IQR)]
        
        ax.set_title(f'Box Plot Analysis: {col}\n'
                    f'IQR: {IQR:.2f}, Outliers: {len(outliers)} ({len(outliers)/len(clean_data)*100:.1f}%)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel(col, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        chart_path = f"{self.charts_folder}/boxplot_{col.replace(' ', '_')}{suffix}.png"
        return self._save_chart(fig, chart_path)
    
    def _create_scatter(self, df: pd.DataFrame, action: Dict, suffix: str) -> Optional[str]:
        """Create enhanced scatter plot with correlation analysis"""
        
        x_col = action.get("x")
        y_col = action.get("y")
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return None
        
        # Clean data
        clean_data = df[[x_col, y_col]].dropna()
        if len(clean_data) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        scatter = ax.scatter(clean_data[x_col], clean_data[y_col], 
                           alpha=0.6, color='coral', s=50)
        
        # Add correlation line if data allows
        try:
            correlation = clean_data[x_col].corr(clean_data[y_col])
            
            # Add trendline
            z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(clean_data[x_col], p(clean_data[x_col]), "r--", alpha=0.8, linewidth=2)
            
            ax.set_title(f'Relationship Analysis: {x_col} vs {y_col}\n'
                        f'Correlation: {correlation:.3f}', 
                        fontsize=14, fontweight='bold', pad=20)
        except:
            ax.set_title(f'Relationship Analysis: {x_col} vs {y_col}', 
                        fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        chart_path = f"{self.charts_folder}/scatter_{x_col.replace(' ', '_')}_vs_{y_col.replace(' ', '_')}{suffix}.png"
        return self._save_chart(fig, chart_path)
    
    def _create_correlation_matrix(self, df: pd.DataFrame, action: Dict, suffix: str) -> Optional[str]:
        """Create enhanced correlation heatmap"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlation
        corr_matrix = numeric_df.corr()
        
        # Create mask for better visualization
        mask = np.triu(np.ones_like(corr_matrix))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True, ax=ax,
                   cbar_kws={"shrink": .8})
        
        ax.set_title('Correlation Analysis Matrix\n'
                    'Strong correlations (|r| > 0.7) highlighted', 
                    fontsize=16, fontweight='bold', pad=20)
        
        chart_path = f"{self.charts_folder}/correlation_matrix{suffix}.png"
        return self._save_chart(fig, chart_path)
    
    def _create_groupby_chart(self, df: pd.DataFrame, action: Dict, suffix: str) -> Optional[str]:
        """Create enhanced groupby analysis chart"""
        
        group_col = action.get("group")
        metric_col = action.get("metric")
        topn = action.get("topn", 10)
        
        if not group_col or not metric_col or group_col not in df.columns or metric_col not in df.columns:
            return None
        
        try:
            # Calculate groupby with additional stats
            grouped = df.groupby(group_col)[metric_col].agg(['sum', 'mean', 'count']).reset_index()
            grouped = grouped.sort_values('sum', ascending=False).head(topn)
            
            if len(grouped) == 0:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Chart 1: Total values
            bars1 = ax1.bar(range(len(grouped)), grouped['sum'], 
                           color='lightgreen', edgecolor='black', alpha=0.7)
            ax1.set_title(f'Top {topn} {group_col} by Total {metric_col}', 
                         fontsize=12, fontweight='bold')
            ax1.set_ylabel(f'Total {metric_col}', fontsize=10, fontweight='bold')
            ax1.set_xticks(range(len(grouped)))
            ax1.set_xticklabels(grouped[group_col], rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}', ha='center', va='bottom', fontsize=8)
            
            # Chart 2: Average values
            bars2 = ax2.bar(range(len(grouped)), grouped['mean'], 
                           color='lightcoral', edgecolor='black', alpha=0.7)
            ax2.set_title(f'Average {metric_col} by {group_col}', 
                         fontsize=12, fontweight='bold')
            ax2.set_ylabel(f'Average {metric_col}', fontsize=10, fontweight='bold')
            ax2.set_xticks(range(len(grouped)))
            ax2.set_xticklabels(grouped[group_col], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            chart_path = f"{self.charts_folder}/groupby_{group_col.replace(' ', '_')}_by_{metric_col.replace(' ', '_')}{suffix}.png"
            return self._save_chart(fig, chart_path)
            
        except Exception as e:
            print(f"    ⚠️ Groupby chart failed: {e}")
            return None
    
    def _create_supply_chain_dashboard(self, df: pd.DataFrame, action: Dict, suffix: str) -> Optional[str]:
        """Create comprehensive supply chain dashboard"""
        
        try:
            # Identify supply chain columns
            supply_cols = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if any(term in col_lower for term in ['wip', 'work in progress']):
                    supply_cols.setdefault('WIP', []).append(col)
                elif any(term in col_lower for term in ['cost', 'value', 'price']):
                    supply_cols.setdefault('Cost/Value', []).append(col)
                elif any(term in col_lower for term in ['qty', 'quantity', 'count']):
                    supply_cols.setdefault('Quantity', []).append(col)
                elif any(term in col_lower for term in ['inventory', 'stock']):
                    supply_cols.setdefault('Inventory', []).append(col)
            
            if not supply_cols:
                return None
            
            # Create dashboard with multiple subplots
            n_charts = len(supply_cols)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            chart_idx = 0
            for category, cols in supply_cols.items():
                if chart_idx >= len(axes):
                    break
                
                ax = axes[chart_idx]
                
                # Calculate total values for each column
                totals = []
                labels = []
                for col in cols[:5]:  # Limit to top 5 columns
                    if df[col].dtype in [np.number]:
                        total = df[col].sum()
                        if pd.notna(total) and total != 0:
                            totals.append(total)
                            labels.append(col)
                
                if totals:
                    bars = ax.bar(range(len(totals)), totals, alpha=0.7)
                    ax.set_title(f'{category} Analysis', fontsize=12, fontweight='bold')
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel('Total Value', fontsize=10)
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:,.0f}', ha='center', va='bottom', fontsize=8)
                
                chart_idx += 1
            
            # Hide unused subplots
            for i in range(chart_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(action.get('title', 'Supply Chain Dashboard'), 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = f"{self.charts_folder}/supply_chain_dashboard{suffix}.png"
            return self._save_chart(fig, chart_path)
            
        except Exception as e:
            print(f"    ⚠️ Supply chain dashboard failed: {e}")
            return None
    
    def _save_chart(self, fig, chart_path: str) -> Optional[str]:
        """Save chart to cloud or local storage"""
        
        try:
            # Save to bytes buffer
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            # Upload to cloud if available
            if upload_bytes:
                upload_bytes(chart_path, buffer.getvalue())
                plt.close(fig)
                return chart_path
            else:
                # Save locally as fallback
                local_path = chart_path.split('/')[-1]
                with open(local_path, 'wb') as f:
                    f.write(buffer.getvalue())
                plt.close(fig)
                return local_path
                
        except Exception as e:
            print(f"    ⚠️ Failed to save chart: {e}")
            plt.close(fig)
            return None
    
    def _generate_action_insight(self, df: pd.DataFrame, action: Dict) -> Optional[str]:
        """Generate business insight for an action"""
        
        action_type = action.get("action")
        
        try:
            if action_type == "histogram":
                col = action.get("column")
                if col in df.columns and df[col].dtype in [np.number]:
                    stats = df[col].describe()
                    skew = df[col].skew()
                    
                    if abs(skew) > 1:
                        skew_desc = "highly skewed" if abs(skew) > 2 else "moderately skewed"
                        skew_dir = "right" if skew > 0 else "left"
                    else:
                        skew_desc = "relatively normal"
                        skew_dir = ""
                    
                    return f"**{col}**: Distribution is {skew_desc} {skew_dir}. Range: {stats['min']:.2f} to {stats['max']:.2f}, Mean: {stats['mean']:.2f}"
            
            elif action_type == "correlation_matrix":
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    strong_corrs = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                strong_corrs.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]} (r={corr_val:.2f})")
                    
                    if strong_corrs:
                        return f"**Strong Correlations Found**: {'; '.join(strong_corrs[:3])}"
                    else:
                        return "**Correlation Analysis**: No strong correlations (|r| > 0.7) found between variables"
            
            elif action_type == "groupby_topn":
                group_col = action.get("group")
                metric_col = action.get("metric")
                
                if group_col in df.columns and metric_col in df.columns:
                    grouped = df.groupby(group_col)[metric_col].sum().sort_values(ascending=False)
                    if len(grouped) > 0:
                        top_value = grouped.iloc[0]
                        top_name = grouped.index[0]
                    else:
                        return f"No data available for grouping by {group_col}"
                    total = grouped.sum()
                    top_percentage = (top_value / total) * 100
                    
                    return f"**Top Performer**: {top_name} accounts for {top_percentage:.1f}% of total {metric_col} (${top_value:,.2f})"
        
        except Exception as e:
            return None
        
        return None
    
    def _ai_driven_followup(self, 
                           df: pd.DataFrame, 
                           sheet_name: str, 
                           previous_rounds: List[Dict], 
                           round_num: int) -> Dict[str, Any]:
        """Generate AI-driven follow-up analysis based on previous rounds"""
        
        if not self.client:
            # Fallback to rule-based follow-up
            return self._rule_based_followup(df, sheet_name, previous_rounds, round_num)
        
        try:
            # Compile insights from previous rounds
            previous_insights = []
            for round_data in previous_rounds:
                insights = round_data.get("insights", [])
                ai_summary = round_data.get("ai_summary")
                if insights:
                    previous_insights.extend(insights)
                if ai_summary:
                    previous_insights.append(f"AI Summary Round {round_data.get('round_number', 0)}: {ai_summary}")
            
            # Create prompt for follow-up suggestions
            context = f"""
Data Analysis Context:
- Dataset: {sheet_name} with {len(df)} rows and {len(df.columns)} columns
- Previous Insights: {'; '.join(previous_insights[:5])}
- Available Columns: {', '.join(df.columns.tolist()[:20])}

Based on the previous analysis, suggest 3-5 deeper EDA actions to uncover business insights.
Focus on: outlier analysis, segment comparisons, trend identification, and root cause exploration.

Return as JSON array:
[{"action": "scatter", "x": "col1", "y": "col2"}, {"action": "histogram", "column": "col3"}]
"""
            
            messages = [{"role": "user", "content": context}]
            response = chat_completion(self.client, messages, model="gpt-4o-mini")
            
            # Parse AI suggestions
            try:
                # Clean and parse JSON
                cleaned_response = response.strip()
                if "```" in cleaned_response:
                    import re
                    cleaned_response = re.sub(r"^```.*?\n", "", cleaned_response, flags=re.MULTILINE)
                    cleaned_response = cleaned_response.replace("```", "").strip()
                
                suggested_actions = json.loads(cleaned_response)
                
                # Execute the AI-suggested round
                return self._execute_eda_round(
                    df, sheet_name, round_num, 
                    analysis_type="ai_followup",
                    suggested_actions=suggested_actions
                )
                
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse AI suggestions, falling back to rule-based")
                return self._rule_based_followup(df, sheet_name, previous_rounds, round_num)
        
        except Exception as e:
            print(f"⚠️ AI follow-up failed: {e}, falling back to rule-based")
            return self._rule_based_followup(df, sheet_name, previous_rounds, round_num)
    
    def _rule_based_followup(self, 
                            df: pd.DataFrame, 
                            sheet_name: str, 
                            previous_rounds: List[Dict], 
                            round_num: int) -> Dict[str, Any]:
        """Rule-based follow-up analysis when AI is not available"""
        
        analysis_type = "deep_dive" if round_num == 2 else "advanced_analysis"
        return self._execute_eda_round(df, sheet_name, round_num, analysis_type)
    
    def _generate_round_ai_summary(self, 
                                  df: pd.DataFrame, 
                                  sheet_name: str, 
                                  round_results: Dict, 
                                  round_num: int) -> Optional[str]:
        """Generate AI summary for a completed round"""
        
        if not self.client:
            return None
        
        try:
            insights = round_results.get("insights", [])
            chart_count = len(round_results.get("chart_paths", []))
            
            context = f"""
Summarize this EDA analysis round for business stakeholders:

Dataset: {sheet_name} ({len(df)} rows, {len(df.columns)} columns)
Round {round_num}: Generated {chart_count} visualizations
Key Findings: {'; '.join(insights[:3])}

Provide a 2-3 sentence business summary focusing on actionable insights and data quality observations.
"""
            
            messages = [{"role": "user", "content": context}]
            summary = chat_completion(self.client, messages, model="gpt-4o-mini")
            return summary.strip()
            
        except Exception as e:
            print(f"⚠️ AI round summary failed: {e}")
            return None
    
    def _generate_business_insights(self, 
                                   df: pd.DataFrame, 
                                   sheet_name: str, 
                                   results: Dict) -> Dict[str, Any]:
        """Generate comprehensive business insights"""
        
        insights = {
            "supply_chain_metrics": {},
            "data_quality_assessment": {},
            "financial_overview": {},
            "operational_insights": {}
        }
        
        # Supply chain specific analysis
        supply_chain_cols = {
            'wip': [col for col in df.columns if 'wip' in str(col).lower()],
            'cost': [col for col in df.columns if any(term in str(col).lower() for term in ['cost', 'value', 'price'])],
            'quantity': [col for col in df.columns if any(term in str(col).lower() for term in ['qty', 'quantity', 'count'])],
            'inventory': [col for col in df.columns if any(term in str(col).lower() for term in ['inventory', 'stock'])]
        }
        
        for category, cols in supply_chain_cols.items():
            if cols:
                try:
                    numeric_cols = [col for col in cols if df[col].dtype in [np.number]]
                    if numeric_cols:
                        total_value = float(df[numeric_cols].sum().sum())
                        avg_value = float(df[numeric_cols].mean().mean())
                        insights["supply_chain_metrics"][category] = {
                            "columns_found": numeric_cols,
                            "total_value": total_value,
                            "average_value": avg_value,
                            "column_count": len(numeric_cols)
                        }
                except:
                    pass
        
        # Data quality assessment
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        insights["data_quality_assessment"] = {
            "missing_data_percentage": round((missing_cells / total_cells) * 100, 2),
            "duplicate_rows": int(duplicate_rows),
            "duplicate_percentage": round((duplicate_rows / len(df)) * 100, 2),
            "completeness_score": round(((total_cells - missing_cells) / total_cells) * 100, 1)
        }
        
        # Financial overview
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            financial_cols = [col for col in numeric_cols if any(term in str(col).lower() 
                                                               for term in ['cost', 'value', 'price', 'amount'])]
            if financial_cols:
                total_financial = float(df[financial_cols].sum().sum())
                insights["financial_overview"] = {
                    "total_financial_value": total_financial,
                    "financial_columns": financial_cols,
                    "average_per_record": total_financial / len(df) if len(df) > 0 else 0
                }
        
        return insights
    
    def _generate_ai_executive_summary(self, 
                                      df: pd.DataFrame, 
                                      sheet_name: str, 
                                      results: Dict) -> Dict[str, Any]:
        """Generate AI-powered executive summary"""
        
        if not self.client:
            return self._generate_basic_summary(df, sheet_name, results)
        
        try:
            # Compile all insights
            all_insights = []
            for round_data in results.get("rounds", []):
                insights = round_data.get("insights", [])
                all_insights.extend(insights)
                
                if round_data.get("ai_summary"):
                    all_insights.append(round_data["ai_summary"])
            
            business_insights = results.get("business_insights", {})
            supply_chain = business_insights.get("supply_chain_metrics", {})
            data_quality = business_insights.get("data_quality_assessment", {})
            
            context = f"""
Create an executive summary for this data analysis:

Dataset: {sheet_name}
Records: {len(df):,} rows, {len(df.columns)} columns
Charts Generated: {len(results.get('chart_paths', []))}

Supply Chain Metrics: {json.dumps(supply_chain, indent=2) if supply_chain else 'None identified'}
Data Quality: {data_quality.get('completeness_score', 0):.1f}% complete, {data_quality.get('missing_data_percentage', 0):.1f}% missing

Key Insights: {'; '.join(all_insights[:5])}

Write a business-focused executive summary covering:
1. Data overview and quality
2. Key financial/operational findings  
3. Supply chain insights (if applicable)
4. Recommended next steps

Keep it professional and actionable for business stakeholders.
"""
            
            messages = [{"role": "user", "content": context}]
            ai_summary = chat_completion(self.client, messages, model="gpt-4o")
            
            return {
                "ai_generated": True,
                "summary_text": ai_summary.strip(),
                "insight_count": len(all_insights),
                "chart_count": len(results.get("chart_paths", [])),
                "data_quality_score": data_quality.get("completeness_score", 0)
            }
            
        except Exception as e:
            print(f"⚠️ AI executive summary failed: {e}")
            return self._generate_basic_summary(df, sheet_name, results)
    
    def _generate_basic_summary(self, 
                               df: pd.DataFrame, 
                               sheet_name: str, 
                               results: Dict) -> Dict[str, Any]:
        """Generate basic summary when AI is not available"""
        
        # Compile insights from all rounds
        all_insights = []
        for round_data in results.get("rounds", []):
            insights = round_data.get("insights", [])
            all_insights.extend(insights)
        
        business_insights = results.get("business_insights", {})
        supply_chain = business_insights.get("supply_chain_metrics", {})
        data_quality = business_insights.get("data_quality_assessment", {})
        
        # Generate basic text summary
        summary_lines = [
            f"Analysis of {sheet_name}: {len(df):,} records with {len(df.columns)} columns",
            f"Data Quality: {data_quality.get('completeness_score', 0):.1f}% complete",
            f"Generated {len(results.get('chart_paths', []))} visualizations across {len(results.get('rounds', []))} analysis rounds"
        ]
        
        if supply_chain:
            summary_lines.append(f"Supply Chain Metrics: {len(supply_chain)} categories identified")
            for category, metrics in supply_chain.items():
                if metrics.get("total_value", 0) > 0:
                    summary_lines.append(f"- {category.title()}: ${metrics['total_value']:,.2f} total value")
        
        if all_insights:
            summary_lines.append("Key Findings:")
            for insight in all_insights[:3]:
                summary_lines.append(f"- {insight}")
        
        return {
            "ai_generated": False,
            "summary_text": "\n".join(summary_lines),
            "insight_count": len(all_insights),
            "chart_count": len(results.get("chart_paths", [])),
            "data_quality_score": data_quality.get("completeness_score", 0)
        }
    
    def _save_comprehensive_results(self, 
                                   results: Dict, 
                                   sheet_name: str, 
                                   filename: str) -> None:
        """Save comprehensive results to cloud storage"""
        
        try:
            # Save JSON results
            results_path = f"{self.summaries_folder}/comprehensive_eda_{sheet_name.replace(' ', '_')}.json"
            
            if upload_json:
                upload_json(results_path, results)
                print(f"📄 Comprehensive results saved: {results_path}")
            else:
                # Save locally as fallback
                local_path = f"comprehensive_eda_{sheet_name.replace(' ', '_')}.json"
                with open(local_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"📄 Comprehensive results saved locally: {local_path}")
            
            # Save executive summary as markdown
            final_summary = results.get("final_summary", {})
            summary_text = final_summary.get("summary_text", "No summary available")
            
            markdown_content = f"""# Executive Summary: {sheet_name}

**Source File**: {filename}  
**Analysis Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Records Analyzed**: {results['metadata']['total_rows']:,}  
**Columns**: {results['metadata']['total_columns']}  
**Charts Generated**: {len(results['chart_paths'])}

## Summary

{summary_text}

## Data Quality Assessment

- **Completeness**: {final_summary.get('data_quality_score', 0):.1f}%
- **Missing Data**: {results['metadata']['missing_percentage']:.1f}%
- **Analysis Rounds**: {len(results['rounds'])}

## Charts Generated

"""
            
            for i, chart_path in enumerate(results['chart_paths'], 1):
                chart_name = chart_path.split('/')[-1]
                markdown_content += f"{i}. `{chart_name}`\n"
            
            markdown_content += "\n## Business Insights\n\n"
            
            business_insights = results.get("business_insights", {})
            supply_chain = business_insights.get("supply_chain_metrics", {})
            
            if supply_chain:
                markdown_content += "### Supply Chain Metrics\n\n"
                for category, metrics in supply_chain.items():
                    if metrics.get("total_value", 0) > 0:
                        markdown_content += f"- **{category.title()}**: ${metrics['total_value']:,.2f} across {metrics.get('column_count', 0)} columns\n"
            
            # Save markdown summary
            summary_path = f"{self.summaries_folder}/executive_summary_{sheet_name.replace(' ', '_')}.md"
            
            if upload_bytes:
                upload_bytes(summary_path, markdown_content.encode('utf-8'))
                print(f"📋 Executive summary saved: {summary_path}")
            else:
                # Save locally as fallback
                local_md_path = f"executive_summary_{sheet_name.replace(' ', '_')}.md"
                with open(local_md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"📋 Executive summary saved locally: {local_md_path}")
                
        except Exception as e:
            print(f"⚠️ Failed to save comprehensive results: {e}")


def run_enhanced_eda(df: pd.DataFrame, 
                    sheet_name: str = "Sheet1",
                    filename: str = "data.xlsx",
                    charts_folder: str = None,
                    summaries_folder: str = None,
                    max_rounds: int = 3) -> Dict[str, Any]:
    """
    Convenience function to run enhanced EDA system.
    
    Args:
        df: DataFrame to analyze
        sheet_name: Name of the sheet/dataset
        filename: Source filename
        charts_folder: Where to save charts (default: /Project_Root/04_Data/02_EDA_Charts)
        summaries_folder: Where to save summaries (default: /Project_Root/04_Data/03_Summaries)
        max_rounds: Maximum number of analysis rounds (default: 3)
    
    Returns:
        Comprehensive results dictionary
    """
    
    eda_system = EnhancedEDASystem(charts_folder, summaries_folder)
    return eda_system.run_comprehensive_eda(df, sheet_name, filename, max_rounds)
