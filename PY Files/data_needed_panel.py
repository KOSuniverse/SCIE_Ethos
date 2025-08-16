# PY Files/data_needed_panel.py
"""
Data Needed Panel for SCIE Ethos Streamlit UI.
Provides collapsible interface for tracking data requirements and gaps.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime

class DataNeededPanel:
    """Panel for tracking data requirements and gaps."""
    
    def __init__(self):
        self.data_gaps = []
        self.data_requirements = []
        self.priority_levels = ["Low", "Medium", "High", "Critical"]
    
    def render_panel(self, expanded: bool = False):
        """Render the data needed panel."""
        with st.expander("📊 Data Needed & Gaps", expanded=expanded):
            self._render_data_gaps()
            self._render_data_requirements()
            self._render_add_gap_form()
    
    def _render_data_gaps(self):
        """Render existing data gaps."""
        if not self.data_gaps:
            st.info("No data gaps identified yet.")
            return
        
        st.markdown("**🔍 Identified Data Gaps:**")
        
        for i, gap in enumerate(self.data_gaps):
            with st.expander(f"Gap {i+1}: {gap.get('description', 'Unknown')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Description:** {gap.get('description', 'N/A')}")
                    st.write(f"**Impact:** {gap.get('impact', 'N/A')}")
                    st.write(f"**Data Type:** {gap.get('data_type', 'N/A')}")
                    st.write(f"**Identified:** {gap.get('timestamp', 'N/A')}")
                
                with col2:
                    priority = gap.get('priority', 'Medium')
                    priority_color = self._get_priority_color(priority)
                    st.markdown(f"**Priority:** {priority_color} {priority}")
                    
                    if st.button(f"Resolve Gap {i+1}", key=f"resolve_{i}"):
                        self._resolve_gap(i)
                
                # Show suggested actions
                actions = gap.get('suggested_actions', [])
                if actions:
                    st.markdown("**Suggested Actions:**")
                    for action in actions:
                        st.markdown(f"- {action}")
    
    def _render_data_requirements(self):
        """Render data requirements."""
        if not self.data_requirements:
            st.info("No specific data requirements identified.")
            return
        
        st.markdown("**📋 Data Requirements:**")
        
        for i, req in enumerate(self.data_requirements):
            with st.expander(f"Requirement {i+1}: {req.get('title', 'Unknown')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Title:** {req.get('title', 'N/A')}")
                    st.write(f"**Description:** {req.get('description', 'N/A')}")
                    st.write(f"**Data Source:** {req.get('data_source', 'N/A')}")
                    st.write(f"**Frequency:** {req.get('frequency', 'N/A')}")
                
                with col2:
                    status = req.get('status', 'Pending')
                    status_color = self._get_status_color(status)
                    st.markdown(f"**Status:** {status_color} {status}")
                    
                    if status == 'Pending':
                        if st.button(f"Mark Complete {i+1}", key=f"complete_{i}"):
                            self._mark_requirement_complete(i)
                
                # Show dependencies
                dependencies = req.get('dependencies', [])
                if dependencies:
                    st.markdown("**Dependencies:**")
                    for dep in dependencies:
                        st.markdown(f"- {dep}")
    
    def _render_add_gap_form(self):
        """Render form to add new data gaps."""
        st.markdown("---")
        st.markdown("**➕ Add New Data Gap:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            description = st.text_area("Description", key="gap_desc", height=80)
            impact = st.text_area("Business Impact", key="gap_impact", height=60)
            data_type = st.selectbox("Data Type", ["Inventory", "WIP", "E&O", "Forecast", "Other"], key="gap_type")
        
        with col2:
            priority = st.selectbox("Priority", self.priority_levels, key="gap_priority")
            suggested_actions = st.text_area("Suggested Actions", key="gap_actions", height=80)
            
            if st.button("Add Gap", key="add_gap"):
                self._add_data_gap(description, impact, data_type, priority, suggested_actions)
                st.success("Data gap added successfully!")
                st.rerun()
    
    def _add_data_gap(self, description: str, impact: str, data_type: str, priority: str, actions: str):
        """Add a new data gap."""
        gap = {
            "description": description,
            "impact": impact,
            "data_type": data_type,
            "priority": priority,
            "suggested_actions": [action.strip() for action in actions.split('\n') if action.strip()],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Open"
        }
        
        self.data_gaps.append(gap)
        
        # Store in session state
        if "data_gaps" not in st.session_state:
            st.session_state.data_gaps = []
        st.session_state.data_gaps.append(gap)
    
    def _resolve_gap(self, gap_index: int):
        """Mark a data gap as resolved."""
        if 0 <= gap_index < len(self.data_gaps):
            self.data_gaps[gap_index]["status"] = "Resolved"
            self.data_gaps[gap_index]["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update session state
            if "data_gaps" in st.session_state:
                st.session_state.data_gaps[gap_index]["status"] = "Resolved"
                st.session_state.data_gaps[gap_index]["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _mark_requirement_complete(self, req_index: int):
        """Mark a data requirement as complete."""
        if 0 <= req_index < len(self.data_requirements):
            self.data_requirements[req_index]["status"] = "Complete"
            self.data_requirements[req_index]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update session state
            if "data_requirements" in st.session_state:
                st.session_state.data_requirements[req_index]["status"] = "Complete"
                st.session_state.data_requirements[req_index]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_priority_color(self, priority: str) -> str:
        """Get color emoji for priority level."""
        priority_colors = {
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🟠",
            "Critical": "🔴"
        }
        return priority_colors.get(priority, "⚪")
    
    def _get_status_color(self, status: str) -> str:
        """Get color emoji for status."""
        status_colors = {
            "Pending": "🟡",
            "In Progress": "🔵",
            "Complete": "🟢",
            "Blocked": "🔴"
        }
        return status_colors.get(status, "⚪")
    
    def add_data_requirement(self, title: str, description: str, data_source: str, frequency: str = "As needed"):
        """Add a new data requirement."""
        requirement = {
            "title": title,
            "description": description,
            "data_source": data_source,
            "frequency": frequency,
            "status": "Pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dependencies": []
        }
        
        self.data_requirements.append(requirement)
        
        # Store in session state
        if "data_requirements" not in st.session_state:
            st.session_state.data_requirements = []
        st.session_state.data_requirements.append(requirement)
    
    def get_gaps_summary(self) -> Dict[str, Any]:
        """Get summary of data gaps for export."""
        open_gaps = [gap for gap in self.data_gaps if gap.get("status") == "Open"]
        resolved_gaps = [gap for gap in self.data_gaps if gap.get("status") == "Resolved"]
        
        return {
            "total_gaps": len(self.data_gaps),
            "open_gaps": len(open_gaps),
            "resolved_gaps": len(resolved_gaps),
            "gaps_by_priority": self._count_by_priority(self.data_gaps),
            "gaps_by_type": self._count_by_type(self.data_gaps),
            "recent_gaps": open_gaps[:5]  # Last 5 open gaps
        }
    
    def _count_by_priority(self, gaps: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count gaps by priority level."""
        counts = {}
        for gap in gaps:
            priority = gap.get("priority", "Unknown")
            counts[priority] = counts.get(priority, 0) + 1
        return counts
    
    def _count_by_type(self, gaps: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count gaps by data type."""
        counts = {}
        for gap in gaps:
            data_type = gap.get("data_type", "Unknown")
            counts[data_type] = counts.get(data_type, 0) + 1
        return counts
    
    def load_from_session(self):
        """Load data gaps and requirements from session state."""
        if "data_gaps" in st.session_state:
            self.data_gaps = st.session_state.data_gaps
        
        if "data_requirements" in st.session_state:
            self.data_requirements = st.session_state.data_requirements
