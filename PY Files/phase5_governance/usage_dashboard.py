# PY Files/phase5_governance/usage_dashboard.py
# Phase 5A: Simple Usage page for Streamlit

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .query_logger import QueryLogger

class UsageDashboard:
    """
    Phase 5A: Simple Usage page with counters and visualizations.
    Displays query statistics, cost tracking, and usage patterns.
    """
    
    def __init__(self):
        self.logger = QueryLogger()
    
    def render_usage_page(self):
        """Render the complete usage dashboard."""
        
        st.title("📊 Usage Dashboard")
        st.markdown("**Phase 5A**: Query logs and usage analytics")
        
        # Time period selector
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            days = st.selectbox(
                "📅 Time Period",
                options=[7, 14, 30, 60, 90],
                index=2,  # Default to 30 days
                format_func=lambda x: f"Last {x} days"
            )
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        with col3:
            export_report = st.button("📥 Export Report")
        
        # Get usage statistics
        stats = self.logger.get_usage_stats(days)
        
        if "error" in stats:
            st.warning(f"⚠️ {stats['error']}")
            st.info("Usage statistics will appear after queries are logged.")
            return
        
        if "message" in stats:
            st.info(f"ℹ️ {stats['message']}")
            return
        
        # Export report if requested
        if export_report:
            try:
                report_path = self.logger.export_usage_report(days)
                st.success(f"✅ Report exported to: `{report_path}`")
                
                # Provide download button
                with open(report_path, "r") as f:
                    report_data = f.read()
                
                st.download_button(
                    label="📥 Download Report",
                    data=report_data,
                    file_name=f"usage_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
        
        # Render dashboard sections
        self._render_summary_metrics(stats)
        self._render_usage_charts(stats)
        self._render_cost_analysis(stats)
        self._render_recent_activity()
    
    def _render_summary_metrics(self, stats: Dict[str, Any]):
        """Render summary metrics cards."""
        
        st.markdown("---")
        st.subheader("📈 Summary Metrics")
        
        summary = stats.get("summary", {})
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Queries",
                value=f"{summary.get('total_queries', 0):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Unique Users",
                value=summary.get('unique_users', 0),
                delta=None
            )
        
        with col3:
            total_cost = summary.get('total_cost_usd', 0)
            st.metric(
                label="Total Cost",
                value=f"${total_cost:.4f}",
                delta=None
            )
        
        with col4:
            avg_response = summary.get('avg_response_time_ms', 0)
            st.metric(
                label="Avg Response Time",
                value=f"{avg_response:.0f}ms",
                delta=None
            )
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tokens = summary.get('total_tokens', 0)
            st.metric(
                label="Total Tokens",
                value=f"{total_tokens:,}",
                delta=None
            )
        
        with col2:
            avg_tokens = summary.get('avg_tokens_per_query', 0)
            st.metric(
                label="Avg Tokens/Query",
                value=f"{avg_tokens:.0f}",
                delta=None
            )
        
        with col3:
            avg_cost = summary.get('avg_cost_per_query', 0)
            st.metric(
                label="Avg Cost/Query",
                value=f"${avg_cost:.6f}",
                delta=None
            )
        
        with col4:
            error_rate = summary.get('error_rate', 0)
            st.metric(
                label="Error Rate",
                value=f"{error_rate:.1f}%",
                delta=None
            )
    
    def _render_usage_charts(self, stats: Dict[str, Any]):
        """Render usage pattern charts."""
        
        st.markdown("---")
        st.subheader("📊 Usage Patterns")
        
        breakdown = stats.get("breakdown", {})
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Intent distribution pie chart
            st.markdown("#### Queries by Intent")
            intent_data = breakdown.get("by_intent", {})
            
            if intent_data:
                df_intent = pd.DataFrame([
                    {"Intent": intent, "Count": count}
                    for intent, count in intent_data.items()
                ])
                
                fig_intent = px.pie(
                    df_intent, 
                    values="Count", 
                    names="Intent",
                    title="Query Distribution by Intent"
                )
                fig_intent.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_intent, use_container_width=True)
            else:
                st.info("No intent data available")
        
        with col2:
            # Model usage bar chart
            st.markdown("#### Queries by Model")
            model_data = breakdown.get("by_model", {})
            
            if model_data:
                df_model = pd.DataFrame([
                    {"Model": model, "Count": count}
                    for model, count in model_data.items()
                ])
                
                fig_model = px.bar(
                    df_model,
                    x="Model",
                    y="Count",
                    title="Query Distribution by Model"
                )
                fig_model.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_model, use_container_width=True)
            else:
                st.info("No model data available")
        
        # Confidence distribution
        st.markdown("#### Confidence Score Distribution")
        confidence_data = breakdown.get("by_confidence", {})
        
        if confidence_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_count = confidence_data.get("High", 0)
                st.metric("🟢 High Confidence", high_count)
            
            with col2:
                med_count = confidence_data.get("Medium", 0)
                st.metric("🟡 Medium Confidence", med_count)
            
            with col3:
                low_count = confidence_data.get("Low", 0)
                st.metric("🔴 Low Confidence", low_count)
            
            # Confidence bar chart
            df_conf = pd.DataFrame([
                {"Confidence": conf, "Count": count}
                for conf, count in confidence_data.items()
            ])
            
            fig_conf = px.bar(
                df_conf,
                x="Confidence",
                y="Count",
                title="Queries by Confidence Level",
                color="Confidence",
                color_discrete_map={"High": "green", "Medium": "orange", "Low": "red"}
            )
            st.plotly_chart(fig_conf, use_container_width=True)
    
    def _render_cost_analysis(self, stats: Dict[str, Any]):
        """Render cost analysis section."""
        
        st.markdown("---")
        st.subheader("💰 Cost Analysis")
        
        summary = stats.get("summary", {})
        breakdown = stats.get("breakdown", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost breakdown by model
            st.markdown("#### Cost Efficiency by Model")
            model_data = breakdown.get("by_model", {})
            total_cost = summary.get("total_cost_usd", 0)
            
            if model_data and total_cost > 0:
                # Estimate cost per model (simplified)
                cost_estimates = []
                for model, count in model_data.items():
                    proportion = count / summary.get("total_queries", 1)
                    estimated_cost = total_cost * proportion
                    cost_estimates.append({
                        "Model": model,
                        "Queries": count,
                        "Estimated Cost": f"${estimated_cost:.6f}",
                        "Cost per Query": f"${estimated_cost/count:.6f}" if count > 0 else "$0"
                    })
                
                df_cost = pd.DataFrame(cost_estimates)
                st.dataframe(df_cost, use_container_width=True)
            else:
                st.info("Cost breakdown not available")
        
        with col2:
            # Token efficiency
            st.markdown("#### Token Efficiency")
            
            total_tokens = summary.get("total_tokens", 0)
            total_queries = summary.get("total_queries", 1)
            avg_tokens = total_tokens / total_queries
            
            efficiency_data = {
                "Metric": [
                    "Total Tokens Used",
                    "Average per Query",
                    "Token Cost Efficiency",
                    "Estimated Monthly Cost"
                ],
                "Value": [
                    f"{total_tokens:,}",
                    f"{avg_tokens:.0f}",
                    f"${total_cost/total_tokens*1000:.4f}/1K tokens" if total_tokens > 0 else "N/A",
                    f"${total_cost * 30:.2f}" if total_cost > 0 else "N/A"
                ]
            }
            
            df_efficiency = pd.DataFrame(efficiency_data)
            st.dataframe(df_efficiency, use_container_width=True, hide_index=True)
    
    def _render_recent_activity(self):
        """Render recent activity section."""
        
        st.markdown("---")
        st.subheader("🕒 Recent Activity")
        
        recent_queries = self.logger.get_recent_queries(15)
        
        if not recent_queries:
            st.info("No recent queries found")
            return
        
        # Prepare data for display
        activity_data = []
        for query in recent_queries:
            activity_data.append({
                "Timestamp": query.get("timestamp", "Unknown")[:19],  # Remove milliseconds
                "User": query.get("user_id", "Unknown"),
                "Intent": query.get("query", {}).get("intent", "Unknown"),
                "Confidence": query.get("response", {}).get("confidence_badge", "Unknown"),
                "Cost": f"${query.get('costs', {}).get('total_cost_usd', 0):.6f}",
                "Tokens": query.get("model", {}).get("total_tokens", 0),
                "Model": query.get("model", {}).get("name", "Unknown"),
                "Response Time": f"{query.get('response', {}).get('response_time_ms', 0)}ms"
            })
        
        df_activity = pd.DataFrame(activity_data)
        st.dataframe(df_activity, use_container_width=True, hide_index=True)
        
        # Show query text for selected row
        if len(activity_data) > 0:
            st.markdown("#### Query Details")
            selected_idx = st.selectbox(
                "Select query to view details:",
                range(len(recent_queries)),
                format_func=lambda x: f"{recent_queries[x].get('timestamp', 'Unknown')[:19]} - {recent_queries[x].get('query', {}).get('intent', 'Unknown')}"
            )
            
            if selected_idx is not None:
                selected_query = recent_queries[selected_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Query Text:**")
                    st.text_area(
                        "Query",
                        value=selected_query.get("query", {}).get("text", "No query text"),
                        height=100,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                with col2:
                    st.markdown("**Sources Used:**")
                    sources = selected_query.get("sources", {})
                    source_info = f"Data sources: {sources.get('source_count', 0)}\n"
                    source_info += f"KB sources: {sources.get('kb_source_count', 0)}\n"
                    
                    if sources.get("kb_sources"):
                        source_info += f"KB docs: {', '.join(sources['kb_sources'][:3])}"
                        if len(sources['kb_sources']) > 3:
                            source_info += f" (+{len(sources['kb_sources']) - 3} more)"
                    
                    st.text_area(
                        "Sources",
                        value=source_info,
                        height=100,
                        disabled=True,
                        label_visibility="collapsed"
                    )
    
    def render_usage_summary_widget(self):
        """Render a compact usage summary widget for sidebar or other locations."""
        
        stats = self.logger.get_usage_stats(7)  # Last 7 days
        
        if "error" in stats or "message" in stats:
            st.info("📊 Usage stats will appear after queries are logged")
            return
        
        summary = stats.get("summary", {})
        
        st.markdown("### 📊 Usage (7 days)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", summary.get("total_queries", 0))
            st.metric("Users", summary.get("unique_users", 0))
        
        with col2:
            st.metric("Cost", f"${summary.get('total_cost_usd', 0):.4f}")
            st.metric("Avg Time", f"{summary.get('avg_response_time_ms', 0):.0f}ms")
        
        # Top intent
        top_intents = stats.get("top_intents", [])
        if top_intents:
            st.markdown(f"**Top Intent**: {top_intents[0]}")
        
        # Most used model
        most_used_model = stats.get("most_used_model", "none")
        st.markdown(f"**Primary Model**: {most_used_model}")
