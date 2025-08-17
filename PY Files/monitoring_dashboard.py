# PY Files/monitoring_dashboard.py
"""
Phase 5: Monitoring Dashboard
Provides real-time system monitoring, performance metrics, and operational insights.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Local imports
try:
    from constants import PROJECT_ROOT
    from path_utils import get_project_paths
    from logging_system import TurnLogger, RetentionManager, AnalyticsEngine
    from qa_framework import QATestRunner
except ImportError:
    PROJECT_ROOT = "/Project_Root"

class MonitoringDashboard:
    """Comprehensive monitoring dashboard for SCIE Ethos system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or PROJECT_ROOT
        
        # Initialize components
        self.logger = TurnLogger(project_root)
        self.retention_manager = RetentionManager(project_root)
        self.analytics_engine = AnalyticsEngine(project_root)
        self.qa_runner = QATestRunner(project_root)
        
        # Dashboard state
        self.refresh_interval = 30  # seconds
        self.last_refresh = datetime.now()
    
    def render_dashboard(self):
        """Render the main monitoring dashboard."""
        st.title("🔍 SCIE Ethos System Monitor")
        st.markdown("Real-time system performance and operational insights")
        
        # Auto-refresh indicator
        self._render_refresh_status()
        
        # Main dashboard sections
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_performance_metrics()
            self._render_confidence_trends()
            self._render_system_health()
        
        with col2:
            self._render_quick_actions()
            self._render_alerts()
            self._render_retention_status()
        
        # Detailed analytics
        self._render_detailed_analytics()
        
        # QA testing section
        self._render_qa_section()
    
    def _render_refresh_status(self):
        """Render refresh status and controls."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.caption(f"Last updated: {self.last_refresh.strftime('%H:%M:%S')}")
        
        with col2:
            if st.button("🔄 Refresh Now"):
                self._refresh_data()
        
        with col3:
            auto_refresh = st.checkbox("Auto-refresh", value=True)
            if auto_refresh:
                time_since_refresh = (datetime.now() - self.last_refresh).total_seconds()
                if time_since_refresh >= self.refresh_interval:
                    self._refresh_data()
    
    def _render_performance_metrics(self):
        """Render key performance metrics."""
        st.subheader("📊 Performance Metrics")
        
        # Get current session data
        session_summary = self.logger.get_session_summary()
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Session Turns",
                value=session_summary.get("turn_count", 0),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Session Duration",
                value=self._format_duration(session_summary.get("started_at")),
                delta=None
            )
        
        with col3:
            # Get confidence trends
            trends = self.logger.get_confidence_trends()
            st.metric(
                label="Avg Confidence",
                value=f"{trends.get('average_confidence', 0.0):.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="System Status",
                value="🟢 Healthy",
                delta=None
            )
    
    def _render_confidence_trends(self):
        """Render confidence trend visualization."""
        st.subheader("📈 Confidence Trends")
        
        try:
            # Get confidence trends from analytics
            trends = self.analytics_engine.analyze_confidence_trends(days=7)
            
            if "error" not in trends:
                # Create confidence trend chart
                fig = go.Figure()
                
                # Add confidence line
                daily_trends = trends.get("daily_trends", {})
                if daily_trends and "confidence" in daily_trends:
                    dates = list(daily_trends["confidence"]["mean"].keys())
                    confidence_values = list(daily_trends["confidence"]["mean"].values())
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=confidence_values,
                        mode='lines+markers',
                        name='Average Confidence',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8)
                    ))
                
                # Update layout
                fig.update_layout(
                    title="7-Day Confidence Trend",
                    xaxis_title="Date",
                    yaxis_title="Confidence Score",
                    yaxis=dict(range=[0, 1]),
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No confidence trend data available yet. Continue using the system to generate data.")
                
        except Exception as e:
            st.error(f"Error rendering confidence trends: {e}")
    
    def _render_system_health(self):
        """Render system health indicators."""
        st.subheader("🏥 System Health")
        
        # Health indicators
        health_metrics = self._get_system_health_metrics()
        
        # Overall health score
        overall_health = health_metrics.get("overall_score", 0.0)
        health_color = "🟢" if overall_health >= 0.8 else "🟡" if overall_health >= 0.6 else "🔴"
        
        st.metric(
            label="Overall Health Score",
            value=f"{health_color} {overall_health:.1%}",
            delta=None
        )
        
        # Health breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Component Health:**")
            for component, score in health_metrics.get("components", {}).items():
                color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                st.caption(f"{color} {component}: {score:.1%}")
        
        with col2:
            st.markdown("**Recent Issues:**")
            issues = health_metrics.get("recent_issues", [])
            if issues:
                for issue in issues[:3]:
                    st.caption(f"⚠️ {issue}")
            else:
                st.caption("✅ No recent issues")
    
    def _render_quick_actions(self):
        """Render quick action buttons."""
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Cleanup Expired Data"):
                with st.spinner("Cleaning up expired data..."):
                    results = self.retention_manager.cleanup_expired_data()
                    st.success(f"Cleanup complete: {results['logs_cleaned']} logs, {results['exports_cleaned']} exports")
            
            if st.button("📊 Generate Report"):
                with st.spinner("Generating performance report..."):
                    report = self.analytics_engine.generate_performance_report(days=30)
                    if "error" not in report:
                        st.success("Performance report generated successfully")
                        st.json(report)
                    else:
                        st.error(f"Error generating report: {report['error']}")
        
        with col2:
            if st.button("🔍 Run QA Tests"):
                with st.spinner("Running QA tests..."):
                    results = self.qa_runner.run_all_tests(use_assistant=True)
                    st.success(f"QA tests complete: {results['test_summary']['passed']}/{results['test_summary']['total']} passed")
            
            if st.button("📈 Export Metrics"):
                self._export_dashboard_metrics()
    
    def _render_alerts(self):
        """Render system alerts and notifications."""
        st.subheader("🚨 Alerts & Notifications")
        
        alerts = self._get_system_alerts()
        
        if not alerts:
            st.success("✅ No active alerts")
            return
        
        for alert in alerts:
            alert_type = alert.get("type", "info")
            message = alert.get("message", "Unknown alert")
            timestamp = alert.get("timestamp", "Unknown time")
            
            if alert_type == "critical":
                st.error(f"🚨 {message}")
            elif alert_type == "warning":
                st.warning(f"⚠️ {message}")
            else:
                st.info(f"ℹ️ {message}")
            
            st.caption(f"Time: {timestamp}")
    
    def _render_retention_status(self):
        """Render data retention status."""
        st.subheader("🗂️ Data Retention")
        
        retention_summary = self.retention_manager.get_retention_summary()
        
        # Policy overview
        st.markdown("**Retention Policy:**")
        policy = retention_summary.get("policy", {})
        
        s3_policy = policy.get("s3", {})
        if "logs" in s3_policy:
            st.caption(f"📝 Logs: {s3_policy['logs']['retention_days']} days")
        if "exports" in s3_policy:
            st.caption(f"📦 Exports: {s3_policy['exports']['retention_days']} days")
        
        # Last cleanup
        last_cleanup = retention_summary.get("last_cleanup", "Unknown")
        next_cleanup = retention_summary.get("next_cleanup_recommended", "Unknown")
        
        st.caption(f"Last cleanup: {last_cleanup}")
        st.caption(f"Next cleanup: {next_cleanup}")
        
        # Cleanup button
        if st.button("🧹 Manual Cleanup"):
            with st.spinner("Running manual cleanup..."):
                results = self.retention_manager.cleanup_expired_data()
                st.success(f"Cleanup complete: {results}")
    
    def _render_detailed_analytics(self):
        """Render detailed analytics section."""
        st.subheader("📊 Detailed Analytics")
        
        # Analytics tabs
        tab1, tab2, tab3 = st.tabs(["Performance", "Usage Patterns", "Quality Metrics"])
        
        with tab1:
            self._render_performance_analytics()
        
        with tab2:
            self._render_usage_patterns()
        
        with tab3:
            self._render_quality_metrics()
    
    def _render_performance_analytics(self):
        """Render performance analytics."""
        st.markdown("**Performance Overview (Last 30 Days)**")
        
        try:
            report = self.analytics_engine.generate_performance_report(days=30)
            
            if "error" not in report:
                trends = report.get("trends", {})
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Total Turns",
                        value=trends.get("total_turns", 0),
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Avg Confidence",
                        value=f"{trends.get('average_confidence', 0.0):.3f}",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        label="Low Confidence Rate",
                        value=f"{trends.get('low_confidence_rate', 0.0):.1%}",
                        delta=None
                    )
                
                # Recommendations
                recommendations = report.get("recommendations", [])
                if recommendations:
                    st.markdown("**Recommendations:**")
                    for rec in recommendations:
                        st.info(rec)
                
                # Alerts
                alerts = report.get("alerts", [])
                if alerts:
                    st.markdown("**Alerts:**")
                    for alert in alerts:
                        st.warning(alert)
            else:
                st.info("No performance data available yet. Continue using the system to generate data.")
                
        except Exception as e:
            st.error(f"Error rendering performance analytics: {e}")
    
    def _render_usage_patterns(self):
        """Render usage pattern analytics."""
        st.markdown("**Usage Pattern Analysis**")
        
        try:
            trends = self.analytics_engine.analyze_confidence_trends(days=30)
            
            if "error" not in trends:
                # Intent distribution
                intent_dist = trends.get("intent_distribution", {})
                if intent_dist:
                    fig = px.pie(
                        values=list(intent_dist.values()),
                        names=list(intent_dist.keys()),
                        title="Intent Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model usage
                model_usage = trends.get("model_usage", {})
                if model_usage:
                    st.markdown("**Model Usage:**")
                    for model, count in model_usage.items():
                        st.caption(f"🤖 {model}: {count} queries")
            else:
                st.info("No usage pattern data available yet.")
                
        except Exception as e:
            st.error(f"Error rendering usage patterns: {e}")
    
    def _render_quality_metrics(self):
        """Render quality metrics."""
        st.markdown("**Quality Metrics**")
        
        try:
            trends = self.analytics_engine.analyze_confidence_trends(days=30)
            
            if "error" not in trends:
                # Quality indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Confidence Distribution:**")
                    avg_conf = trends.get("average_confidence", 0.0)
                    conf_std = trends.get("confidence_std", 0.0)
                    
                    st.metric("Average", f"{avg_conf:.3f}")
                    st.metric("Std Dev", f"{conf_std:.3f}")
                
                with col2:
                    st.markdown("**Quality Thresholds:**")
                    low_conf_rate = trends.get("low_confidence_rate", 0.0)
                    
                    if low_conf_rate < 0.1:
                        st.success(f"Low confidence rate: {low_conf_rate:.1%}")
                    elif low_conf_rate < 0.3:
                        st.warning(f"Low confidence rate: {low_conf_rate:.1%}")
                    else:
                        st.error(f"Low confidence rate: {low_conf_rate:.1%}")
                
                # Quality trends
                daily_trends = trends.get("daily_trends", {})
                if daily_trends and "confidence" in daily_trends:
                    st.markdown("**Daily Quality Trends:**")
                    
                    # Create quality trend chart
                    dates = list(daily_trends["confidence"]["mean"].keys())
                    confidence_values = list(daily_trends["confidence"]["mean"].values())
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=confidence_values,
                        mode='lines+markers',
                        name='Daily Confidence',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Daily Quality Trends",
                        xaxis_title="Date",
                        yaxis_title="Confidence Score",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No quality metrics available yet.")
                
        except Exception as e:
            st.error(f"Error rendering quality metrics: {e}")
    
    def _render_qa_section(self):
        """Render QA testing section."""
        st.subheader("🧪 QA Testing & Validation")
        
        # QA test controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**Acceptance Test Suite**")
            st.caption("Run comprehensive QA tests to validate system performance")
        
        with col2:
            if st.button("▶️ Run Tests"):
                self._run_qa_tests()
        
        with col3:
            if st.button("📋 View Results"):
                self._view_qa_results()
        
        # Recent test results
        self._render_recent_qa_results()
    
    def _run_qa_tests(self):
        """Run QA tests and display results."""
        with st.spinner("Running QA acceptance tests..."):
            try:
                results = self.qa_runner.run_all_tests(use_assistant=True)
                
                # Display results
                st.success(f"QA Tests Complete!")
                
                # Summary metrics
                summary = results.get("test_summary", {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Tests", summary.get("total_tests", 0))
                
                with col2:
                    st.metric("Passed", summary.get("passed", 0))
                
                with col3:
                    st.metric("Overall Score", f"{summary.get('overall_score', 0.0):.1%}")
                
                # Recommendations
                recommendations = results.get("recommendations", [])
                if recommendations:
                    st.markdown("**Recommendations:**")
                    for rec in recommendations:
                        st.info(rec)
                
                # Save results
                report_path = self.qa_runner.save_test_report(results)
                if report_path:
                    st.success(f"Test report saved to: {report_path}")
                
            except Exception as e:
                st.error(f"Error running QA tests: {e}")
    
    def _view_qa_results(self):
        """View recent QA test results."""
        try:
            # Look for recent QA reports
            reports_dir = Path(self.project_root) / "06_Logs" / "qa_reports"
            if not reports_dir.exists():
                st.info("No QA reports found")
                return
            
            # Find most recent report
            report_files = list(reports_dir.glob("qa_test_report_*.json"))
            if not report_files:
                st.info("No QA reports found")
                return
            
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            
            # Load and display report
            with open(latest_report, 'r') as f:
                report = json.load(f)
            
            st.success(f"Latest QA Report: {latest_report.name}")
            
            # Display summary
            summary = report.get("test_summary", {})
            st.json(summary)
            
        except Exception as e:
            st.error(f"Error viewing QA results: {e}")
    
    def _render_recent_qa_results(self):
        """Render recent QA test results."""
        st.markdown("**Recent Test Results**")
        
        try:
            # Look for recent QA reports
            reports_dir = Path(self.project_root) / "06_Logs" / "qa_reports"
            if not reports_dir.exists():
                st.info("No recent QA test results available")
                return
            
            # Find recent reports
            report_files = list(reports_dir.glob("qa_test_report_*.json"))
            if not report_files:
                st.info("No recent QA test results available")
                return
            
            # Sort by modification time
            report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Display recent results
            for report_file in report_files[:3]:  # Show last 3 reports
                try:
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                    
                    summary = report.get("test_summary", {})
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.caption(f"📋 {report_file.name}")
                    
                    with col2:
                        passed = summary.get("passed", 0)
                        total = summary.get("total_tests", 0)
                        st.caption(f"{passed}/{total}")
                    
                    with col3:
                        score = summary.get("overall_score", 0.0)
                        st.caption(f"{score:.1%}")
                        
                except Exception as e:
                    st.caption(f"Error reading {report_file.name}: {e}")
                    
        except Exception as e:
            st.error(f"Error rendering recent QA results: {e}")
    
    def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics."""
        try:
            # Analyze recent performance
            trends = self.analytics_engine.analyze_confidence_trends(days=7)
            
            if "error" in trends:
                return {
                    "overall_score": 0.5,
                    "components": {
                        "logging": 0.5,
                        "analytics": 0.5,
                        "qa": 0.5
                    },
                    "recent_issues": ["Limited data for analysis"]
                }
            
            # Calculate health scores
            avg_confidence = trends.get("average_confidence", 0.5)
            low_conf_rate = trends.get("low_confidence_rate", 0.5)
            total_turns = trends.get("total_turns", 0)
            
            # Component health scores
            logging_health = 0.9 if total_turns > 0 else 0.5
            analytics_health = 0.8 if "error" not in trends else 0.3
            qa_health = 0.7  # Placeholder
            
            # Overall health score
            overall_score = (logging_health + analytics_health + qa_health) / 3
            
            # Recent issues
            recent_issues = []
            if low_conf_rate > 0.3:
                recent_issues.append("High rate of low confidence responses")
            if total_turns < 5:
                recent_issues.append("Limited system usage data")
            
            return {
                "overall_score": overall_score,
                "components": {
                    "logging": logging_health,
                    "analytics": analytics_health,
                    "qa": qa_health
                },
                "recent_issues": recent_issues
            }
            
        except Exception as e:
            return {
                "overall_score": 0.3,
                "components": {
                    "logging": 0.3,
                    "analytics": 0.3,
                    "qa": 0.3
                },
                "recent_issues": [f"System error: {e}"]
            }
    
    def _get_system_alerts(self) -> List[Dict[str, Any]]:
        """Get system alerts and notifications."""
        alerts = []
        
        try:
            # Check confidence trends
            trends = self.analytics_engine.analyze_confidence_trends(days=7)
            
            if "error" not in trends:
                avg_confidence = trends.get("average_confidence", 0.5)
                low_conf_rate = trends.get("low_confidence_rate", 0.5)
                
                if avg_confidence < 0.5:
                    alerts.append({
                        "type": "critical",
                        "message": f"Average confidence below 0.5: {avg_confidence:.3f}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                elif avg_confidence < 0.7:
                    alerts.append({
                        "type": "warning",
                        "message": f"Average confidence below 0.7: {avg_confidence:.3f}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                
                if low_conf_rate > 0.5:
                    alerts.append({
                        "type": "critical",
                        "message": f"High rate of low confidence responses: {low_conf_rate:.1%}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                elif low_conf_rate > 0.3:
                    alerts.append({
                        "type": "warning",
                        "message": f"Elevated rate of low confidence responses: {low_conf_rate:.1%}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
            
            # Check retention policy
            retention_summary = self.retention_manager.get_retention_summary()
            last_cleanup = retention_summary.get("last_cleanup")
            
            if last_cleanup:
                try:
                    last_cleanup_dt = datetime.fromisoformat(last_cleanup.replace("Z", "+00:00"))
                    days_since_cleanup = (datetime.now() - last_cleanup_dt).days
                    
                    if days_since_cleanup > 7:
                        alerts.append({
                            "type": "warning",
                            "message": f"Data cleanup overdue by {days_since_cleanup} days",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                except:
                    pass
            
        except Exception as e:
            alerts.append({
                "type": "critical",
                "message": f"Error checking system alerts: {e}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
        return alerts
    
    def _refresh_data(self):
        """Refresh dashboard data."""
        self.last_refresh = datetime.now()
        st.rerun()
    
    def _format_duration(self, start_time_str: str) -> str:
        """Format duration since start time."""
        if not start_time_str:
            return "Unknown"
        
        try:
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            duration = datetime.now() - start_time
            
            if duration.days > 0:
                return f"{duration.days}d {duration.seconds // 3600}h"
            elif duration.seconds > 3600:
                return f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
            else:
                return f"{duration.seconds // 60}m"
        except:
            return "Unknown"
    
    def _export_dashboard_metrics(self):
        """Export dashboard metrics."""
        try:
            # Collect metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "session_summary": self.logger.get_session_summary(),
                "system_health": self._get_system_health_metrics(),
                "retention_status": self.retention_manager.get_retention_summary()
            }
            
            # Export to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_metrics_{timestamp}.json"
            
            export_dir = Path(self.project_root) / "06_Logs" / "dashboard_exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            export_path = export_dir / filename
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, default=str, ensure_ascii=False)
            
            st.success(f"Dashboard metrics exported to: {export_path}")
            
        except Exception as e:
            st.error(f"Error exporting dashboard metrics: {e}")
