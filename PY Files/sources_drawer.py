# PY Files/sources_drawer.py
"""
Enhanced Sources Drawer for SCIE Ethos Streamlit UI.
Provides clickable citations and source management.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import json
import time

class SourcesDrawer:
    """Enhanced sources drawer with clickable citations and source management."""
    
    def __init__(self):
        self.expanded = False
        self.data_gaps = []  # Add this for test compatibility
    
    def render_sources_panel(self, sources: Dict[str, Any], confidence_score: float = None):
        """Render the sources panel with enhanced functionality."""
        
        # Sources header with confidence
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📚 Sources & Citations")
        with col2:
            if confidence_score is not None:
                self._render_confidence_badge(confidence_score)
        
        # Sources content
        if not sources:
            st.info("No sources available for this response.")
            return
        
        # File sources (Assistant files)
        file_sources = sources.get("file_sources", [])
        if file_sources:
            st.markdown("**📁 Assistant Files:**")
            for source in file_sources:
                self._render_file_source(source)
        
        # Knowledge base sources
        kb_sources = sources.get("kb_sources", [])
        if kb_sources:
            st.markdown("**📚 Knowledge Base:**")
            for source in kb_sources:
                self._render_kb_source(source)
        
        # Data sources (Excel/CSV files)
        data_sources = sources.get("data_sources", [])
        if data_sources:
            st.markdown("**📊 Data Sources:**")
            for source in data_sources:
                self._render_data_source(source)
        
        # External sources
        external_sources = sources.get("external_sources", [])
        if external_sources:
            st.markdown("**🌐 External Sources:**")
            for source in external_sources:
                self._render_external_source(source)
        
        # Source metadata
        source_metadata = sources.get("metadata", {})
        if source_metadata:
            st.markdown("**📋 Source Metadata:**")
            for key, value in source_metadata.items():
                st.caption(f"**{key}:** {value}")
    
    def _render_confidence_badge(self, score: float):
        """Render confidence badge with appropriate styling."""
        if score >= 0.8:
            badge_color = "🟢"
            level = "HIGH"
        elif score >= 0.6:
            badge_color = "🟡"
            level = "MED"
        else:
            badge_color = "🔴"
            level = "LOW"
        
        st.markdown(f"{badge_color} **{level}** ({score:.2f})")
    
    def _render_file_source(self, source: Dict[str, Any]):
        """Render a file source with clickable functionality."""
        filename = source.get("filename", "Unknown")
        file_id = source.get("file_id", "")
        file_type = source.get("file_type", "unknown")
        relevance_score = source.get("relevance_score", 0)
        
        # Create a clickable source card
        with st.expander(f"📄 {filename} ({file_type.upper()})", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**File ID:** {file_id}")
                st.write(f"**Type:** {file_type}")
                if relevance_score > 0:
                    st.write(f"**Relevance:** {relevance_score:.2f}")
            
            with col2:
                if st.button(f"📥 Download {filename}", key=f"dl_{file_id}"):
                    self._download_file(file_id, filename)
            
            # Show file content preview if available
            content_preview = source.get("content_preview", "")
            if content_preview:
                st.markdown("**Content Preview:**")
                st.text(content_preview[:500] + "..." if len(content_preview) > 500 else content_preview)
    
    def _render_kb_source(self, source: Dict[str, Any]):
        """Render a knowledge base source."""
        doc_title = source.get("title", "Unknown Document")
        doc_type = source.get("doc_type", "unknown")
        doc_path = source.get("path", "")
        confidence = source.get("confidence", 0)
        
        with st.expander(f"📚 {doc_title} ({doc_type})", expanded=False):
            st.write(f"**Path:** {doc_path}")
            st.write(f"**Confidence:** {confidence:.2f}")
            
            # Show document summary if available
            summary = source.get("summary", "")
            if summary:
                st.markdown("**Summary:**")
                st.write(summary)
            
            # Show tags if available
            tags = source.get("tags", [])
            if tags:
                st.markdown("**Tags:**")
                for tag in tags:
                    st.markdown(f"- `{tag}`")
    
    def _render_data_source(self, source: Dict[str, Any]):
        """Render a data source (Excel/CSV)."""
        filename = source.get("filename", "Unknown")
        sheet_name = source.get("sheet_name", "")
        row_count = source.get("row_count", 0)
        col_count = source.get("col_count", 0)
        data_type = source.get("data_type", "unknown")
        
        with st.expander(f"📊 {filename} - {sheet_name}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**File:** {filename}")
                st.write(f"**Sheet:** {sheet_name}")
            with col2:
                st.write(f"**Rows:** {row_count:,}")
                st.write(f"**Columns:** {col_count}")
            
            st.write(f"**Type:** {data_type}")
            
            # Show data preview if available
            preview_data = source.get("preview_data", None)
            if preview_data:
                st.markdown("**Data Preview:**")
                st.dataframe(preview_data, use_container_width=True)
    
    def _render_external_source(self, source: Dict[str, Any]):
        """Render an external source."""
        title = source.get("title", "Unknown")
        url = source.get("url", "")
        source_type = source.get("source_type", "unknown")
        timestamp = source.get("timestamp", "")
        
        with st.expander(f"🌐 {title}", expanded=False):
            st.write(f"**Type:** {source_type}")
            if timestamp:
                st.write(f"**Timestamp:** {timestamp}")
            
            if url:
                st.write(f"**URL:** {url}")
                if st.button(f"🔗 Open {title}", key=f"open_{hash(url)}"):
                    st.markdown(f"[Open {title}]({url})")
    
    def _download_file(self, file_id: str, filename: str):
        """Handle file download (placeholder for now)."""
        st.info(f"Download functionality for {filename} would be implemented here.")
        # TODO: Implement actual file download from OpenAI Assistant
    
    def render_collapsible_sources(self, sources: Dict[str, Any], confidence_score: float = None):
        """Render sources in a collapsible panel."""
        with st.expander("📚 Sources & Citations", expanded=False):
            self.render_sources_panel(sources, confidence_score)
    
    def render_inline_sources(self, sources: Dict[str, Any], confidence_score: float = None):
        """Render sources inline with the response."""
        if not sources:
            return
        
        st.markdown("---")
        st.markdown("**📚 Sources:**")
        
        # Show key sources inline
        file_sources = sources.get("file_sources", [])
        if file_sources:
            source_list = [f"📄 {s.get('filename', 'Unknown')}" for s in file_sources[:3]]
            if len(file_sources) > 3:
                source_list.append(f"... and {len(file_sources) - 3} more")
            st.markdown(" • ".join(source_list))
        
        # Show confidence if available
        if confidence_score is not None:
            st.caption(f"Confidence: {confidence_score:.2f}")
        
        # Expandable full sources
        with st.expander("View all sources", expanded=False):
            self.render_sources_panel(sources, confidence_score)
