# chat_ui.py — Architecture-Compliant Streamlit Chat Interface

import os
import json
import time
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Make local modules importable
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(str((Path(__file__).resolve().parent / "PY Files").resolve()))

# Core enterprise modules
from constants import PROJECT_ROOT, META_DIR, DATA_ROOT, CLEANSED
from session import SessionState
from logger import log_event, log_query_result
from orchestrator import answer_question
from assistant_bridge import auto_model, run_query, run_query_with_files
from confidence import score_ravc, should_abstain, score_confidence_enhanced, get_service_level_zscore, get_confidence_badge
from path_utils import get_project_paths
from dbx_utils import list_data_files

# Phase 4 components
from export_utils import ExportManager
from sources_drawer import SourcesDrawer
from data_needed_panel import DataNeededPanel

# Set page configuration first (only when run as standalone)
if __name__ == "__main__":
    st.set_page_config(
        page_title="SCIE Ethos Analyst", 
        page_icon="🧠", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_chat_assistant():
    """Render the complete chat assistant UI inline."""
    
    # Custom CSS for ChatGPT-like appearance
    st.markdown("""
    <style>
        .stChatMessage {
            font-size: 1rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .badge {
            padding: 4px 8px;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin: 2px;
        }
        .confidence-high { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }
        .confidence-medium { background: linear-gradient(135deg, #FF9800 0%, #f57c00 100%); }
        .confidence-low { background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%); }
        .model-mini { background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); }
        .model-4o { background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%); }
        .model-assistant { background: linear-gradient(135deg, #00BCD4 0%, #0097A7 100%); }
        .sources-card {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
        .service-level-badge {
            background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            display: inline-block;
            margin: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # Session State Initialization
    # ─────────────────────────────────────────────────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = "default"
    if "session_handler" not in st.session_state:
        st.session_state.session_handler = SessionState()
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = {}
    if "confidence_history" not in st.session_state:
        st.session_state.confidence_history = []
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "service_level" not in st.session_state:
        st.session_state.service_level = 0.95
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    # ─────────────────────────────────────────────────────────────────────────────
    # Helper Functions
    # ─────────────────────────────────────────────────────────────────────────────
    def get_service_level_badge(service_level: float) -> str:
        """Generate service level badge with z-score."""
        z_score = get_service_level_zscore(service_level)
        return f'<span class="service-level-badge">{service_level:.1%} (z={z_score:.3f})</span>'

    # ─────────────────────────────────────────────────────────────────────────────
    # Sidebar Configuration
    # ─────────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🧠 SCIE Ethos Control Panel")
        
        # Service Level Control (Auto-set to 95% - no dropdowns)
        st.subheader("⚙️ Service Level")
        service_level = 0.95  # Fixed at 95% - no user selection
        
        # Update session state
        if service_level != st.session_state.service_level:
            st.session_state.service_level = service_level
            st.rerun()
        
        # Display service level badge
        service_level_badge = get_service_level_badge(service_level)
        st.markdown(f"**Current:** {service_level_badge}", unsafe_allow_html=True)
        
        # Z-score explanation
        z_score = get_service_level_zscore(service_level)
        st.caption(f"Z-score: {z_score:.3f} (confidence interval)")
        
        st.markdown("---")
        
        # Model Selection (Auto - no user selection)
        st.subheader("🤖 Model Selection")
        model_choice = "Auto (Recommended)"  # Fixed - no user selection
        
        # Confidence History
        if st.session_state.confidence_history:
            st.subheader("📊 Confidence History")
            st.line_chart(st.session_state.confidence_history)
        
        # Data Needed Panel (Always rendered as per spec)
        st.markdown("---")
        st.subheader("📊 Data Needed & Gaps")
        
        # Initialize data needed panel
        if "data_needed_panel" not in st.session_state:
            st.session_state.data_needed_panel = DataNeededPanel()
        
        data_panel = st.session_state.data_needed_panel
        data_panel.load_from_session()
        
        # Render data needed panel in sidebar
        data_panel.render_panel(expanded=False)

    # ─────────────────────────────────────────────────────────────────────────────
    # Main Chat Interface
    # ─────────────────────────────────────────────────────────────────────────────
    st.title("🧠 SCIE Ethos LLM Assistant")
    st.caption("Architecture-Compliant Chat Interface with Enhanced Phase 4 Features")

    # Confidence badge (from last interaction)
    if st.session_state.confidence_history:
        last_confidence = st.session_state.confidence_history[-1]
        # Ensure confidence is a valid float
        try:
            if isinstance(last_confidence, dict):
                confidence_value = last_confidence.get("score", 0.5)
            else:
                confidence_value = float(last_confidence) if last_confidence is not None else 0.5
            confidence_badge_html = get_confidence_badge(confidence_value, st.session_state.service_level)
            st.markdown(f"**Confidence:** {confidence_badge_html}", unsafe_allow_html=True)
        except (ValueError, TypeError) as e:
            # Fallback if confidence formatting fails
            st.markdown("**Confidence:** Medium (0.50)", unsafe_allow_html=True)

    # Chat messages display
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show artifacts if available
            if message.get("artifacts"):
                st.markdown("**📎 Generated Artifacts:**")
                for artifact in message["artifacts"]:
                    st.code(str(artifact))

    # Chat input
    if prompt := st.chat_input("Ask about inventory, WIP, E&O, forecasting, or root cause analysis..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response placeholder
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Process the query
            try:
                # Initialize export manager with current service level
                export_manager = ExportManager(st.session_state.service_level)
                
                # Get AI response (working)
                if st.session_state.selected_files:
                    ai_response = run_query_with_files(
                        prompt,
                        st.session_state.selected_files,
                        thread_id=st.session_state.thread_id
                    )
                else:
                    ai_response = run_query(
                        prompt,
                        thread_id=st.session_state.thread_id
                    )
                
                # Try to get KB document answer alongside AI
                kb_answer = "No relevant documents found in knowledge base."
                kb_sources = []
                
                try:
                    from dropbox_kb_sync import list_kb_candidates, get_folder_structure
                    
                    # Get KB files
                    kb_path = os.getenv("KB_DBX_PATH", "/Project_Root/06_LLM_Knowledge_Base")
                    data_path = os.getenv("DATA_DBX_PATH", "/Project_Root/04_Data")
                    
                    candidates = list_kb_candidates(kb_path, data_path)
                    kb_files = candidates.get('kb_docs', [])
                    data_files = candidates.get('data_files', [])
                    
                    # Add data files to search scope
                    all_files = kb_files + data_files
                    
                    if all_files:
                        # Enhanced relevance check with folder prioritization
                        keywords = prompt.lower().split()
                        # Add special cases for common terms
                        if "s&op" in prompt.lower() or "siop" in prompt.lower():
                            keywords.extend(["siop", "s&op", "sales", "operations", "planning", "quarterly"])
                        if "launch" in prompt.lower():
                            keywords.extend(["launch", "launching", "product"])
                        if "wip" in prompt.lower() or "work in progress" in prompt.lower():
                            keywords.extend(["wip", "work", "progress", "inventory", "manufacturing"])
                        if "inventory" in prompt.lower():
                            keywords.extend(["inventory", "stock", "on hand", "oh"])
                        if any(word in prompt.lower() for word in ["us", "united states", "america"]):
                            keywords.extend(["us", "usa", "america", "united", "states"])
                        
                        # Define folder priorities (1 = highest priority)
                        # Boost data file priority for data-specific queries
                        is_data_query = any(term in prompt.lower() for term in ["wip", "inventory", "stock", "data", "numbers", "chart", "table"])
                        
                        folder_priorities = {
                            "meeting minutes": 1,
                            "meetings": 1, 
                            "live events": 2,
                            "quarterly": 2,
                            "email": 2,
                            "data": 1 if is_data_query else 3,  # Higher priority for data queries
                            "cleansed": 1 if is_data_query else 3,  # Data folder variants
                            "training": 4,
                            "apics": 4,
                            "source_docs": 5,
                            "root": 6
                        }
                        
                        scored_files = []
                        
                        for file_info in all_files[:15]:  # Check more files
                            file_name = file_info['name'].lower()
                            folder_name = file_info.get('folder', '').lower()
                            
                            # Calculate relevance score
                            relevance_score = 0
                            keyword_matches = 0
                            
                            # Check keyword matching
                            for keyword in keywords:
                                if len(keyword) > 2:
                                    if keyword in file_name:
                                        keyword_matches += 2  # Higher weight for filename matches
                                    elif keyword in folder_name:
                                        keyword_matches += 1
                            
                            if keyword_matches > 0:
                                # Get folder priority (lower number = higher priority)
                                folder_priority = 6  # Default lowest priority
                                for folder_key, priority in folder_priorities.items():
                                    if folder_key in folder_name:
                                        folder_priority = min(folder_priority, priority)
                                        break
                                
                                # Calculate final score (higher = better)
                                relevance_score = keyword_matches * 10 - folder_priority
                                
                                # Bonus for exact matches or multiple keywords
                                if any(keyword in file_name for keyword in keywords if len(keyword) > 4):
                                    relevance_score += 5
                                
                                scored_files.append({
                                    **file_info,
                                    'relevance_score': relevance_score,
                                    'keyword_matches': keyword_matches,
                                    'folder_priority': folder_priority
                                })
                        
                        # Sort by relevance score (highest first)
                        relevant_files = sorted(scored_files, key=lambda x: x['relevance_score'], reverse=True)
                        
                        if relevant_files:
                            kb_sources = [{"name": f["name"], "folder": f.get("folder", "root")} for f in relevant_files[:5]]
                            
                            # Try to upload and get content from the most relevant files
                            try:
                                from dbx_utils import upload_dropbox_file_to_openai
                                from openai import OpenAI
                                
                                client = OpenAI()
                                uploaded_files = []
                                
                                # Upload top relevant files with priority weighting (filter out unsupported formats)
                                supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.pptx'}
                                
                                # Separate by priority groups
                                high_priority = [f for f in relevant_files if f.get('folder_priority', 6) <= 2]
                                medium_priority = [f for f in relevant_files if f.get('folder_priority', 6) == 3]
                                low_priority = [f for f in relevant_files if f.get('folder_priority', 6) > 3]
                                
                                # Try to upload from each priority group
                                files_to_try = (high_priority[:2] + medium_priority[:2] + low_priority[:1])[:4]
                                
                                for file_info in files_to_try:
                                    file_ext = os.path.splitext(file_info['name'])[1].lower()
                                    
                                    if file_ext in supported_extensions:
                                        try:
                                            file_id = upload_dropbox_file_to_openai(file_info['path'], purpose="assistants")
                                            if file_id:
                                                uploaded_files.append({
                                                    "file_id": file_id, 
                                                    "name": file_info['name'],
                                                    "folder": file_info.get('folder', 'root'),
                                                    "priority": file_info.get('folder_priority', 6),
                                                    "score": file_info.get('relevance_score', 0)
                                                })
                                                if len(uploaded_files) >= 3:  # Upload up to 3 files
                                                    break
                                        except Exception as e:
                                            print(f"Failed to upload {file_info['name']}: {e}")
                                    else:
                                        print(f"Skipping {file_info['name']} - unsupported format ({file_ext})")
                                
                                if uploaded_files:
                                    # Create a thread and ask for summary
                                    thread = client.beta.threads.create()
                                    
                                    # Get assistant ID
                                    try:
                                        with open("prompts/assistant.json", "r", encoding="utf-8") as f:
                                            meta = json.load(f)
                                            assistant_id = meta["assistant_id"]
                                    except:
                                        assistant_id = os.getenv("ASSISTANT_ID")
                                    
                                    if assistant_id:
                                        # Create detailed prompt for multi-document analysis
                                        file_list = "\n".join([f"- {f['name']} (from {f['folder']} folder, priority {f['priority']})" for f in uploaded_files])
                                        
                                        analysis_prompt = f"""I have uploaded {len(uploaded_files)} documents to analyze. Please answer this question: {prompt}

UPLOADED DOCUMENTS:
{file_list}

INSTRUCTIONS:
1. Analyze ALL uploaded documents to find relevant information
2. For each piece of information, use this format: "From [DOCUMENT NAME] (folder: [FOLDER]): [information]"
3. If documents have conflicting information, mention both viewpoints
4. Prioritize information from meeting minutes and live events over training materials
5. For data files (.xlsx), look for specific tables, numbers, and data points
6. Provide specific details, quotes, data points, and table information where available
7. If no relevant information is found in the documents, clearly state this
8. Do NOT use OpenAI's internal citation format (【4:1†source】) - use the clear format specified above

Please provide a comprehensive analysis that synthesizes information from all relevant documents with clear source attribution."""

                                        client.beta.threads.messages.create(
                                            thread_id=thread.id,
                                            role="user",
                                            content=analysis_prompt
                                        )
                                        
                                        # Run the assistant
                                        run = client.beta.threads.runs.create(
                                            thread_id=thread.id,
                                            assistant_id=assistant_id
                                        )
                                        
                                        # Wait for completion (simple polling)
                                        max_wait = 30  # 30 second timeout
                                        waited = 0
                                        while run.status not in ["completed", "failed", "cancelled", "expired"] and waited < max_wait:
                                            time.sleep(2)
                                            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                                            waited += 2
                                        
                                        if run.status == "completed":
                                            # Get the response
                                            messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")
                                            if messages.data:
                                                latest_message = messages.data[0]
                                                if hasattr(latest_message, 'content') and latest_message.content:
                                                    content_block = latest_message.content[0]
                                                    if hasattr(content_block, 'text'):
                                                        kb_answer = content_block.text.value
                                                    else:
                                                        kb_answer = f"Found {len(relevant_files)} relevant documents but could not extract summary."
                                                else:
                                                    kb_answer = f"Found {len(relevant_files)} relevant documents but received empty response."
                                        else:
                                            kb_answer = f"Found {len(relevant_files)} relevant documents but processing timed out."
                                    else:
                                        kb_answer = f"Found {len(relevant_files)} relevant documents but no assistant configured."
                                else:
                                    kb_answer = f"Found {len(relevant_files)} relevant documents but could not process them:\n"
                                    for f in relevant_files[:3]:
                                        file_ext = os.path.splitext(f['name'])[1].lower()
                                        status = "unsupported format" if file_ext not in supported_extensions else "upload failed"
                                        kb_answer += f"• {f['name']} (in {f.get('folder', 'root')}) - {status}\n"
                                        
                            except Exception as e:
                                kb_answer = f"Found {len(relevant_files)} relevant documents but could not process them: {str(e)}\n"
                                for f in relevant_files[:3]:
                                    kb_answer += f"• {f['name']} (in {f.get('folder', 'root')})\n"
                        else:
                            kb_answer = "No documents found matching your query keywords."
                    
                except Exception as e:
                    kb_answer = f"Knowledge base search unavailable: {str(e)}"
                
                # Combine responses
                ai_answer = ai_response.get("answer", "No AI response available")
                
                # Create dual format answer
                answer = f"""### 📚 Knowledge Base Answer
{kb_answer}

### 🤖 AI Analysis
{ai_answer}"""
                
                # Combine sources
                ai_sources = ai_response.get("sources", {})
                sources = {
                    "kb_sources": kb_sources,
                    "ai_sources": ai_sources
                }
                
                confidence_raw = ai_response.get("confidence", 0.5)
                
                # Handle confidence score - it might be a dict from score_ravc() or a float
                if isinstance(confidence_raw, dict):
                    confidence_score = confidence_raw.get("score", 0.5)
                else:
                    confidence_score = float(confidence_raw) if confidence_raw is not None else 0.5
                
                # Capture thread_id for conversation continuity (from AI response)
                if ai_response.get("thread_id") and not st.session_state.thread_id:
                    st.session_state.thread_id = ai_response["thread_id"]
                
                # Update confidence history
                st.session_state.confidence_history.append(confidence_score)
                if len(st.session_state.confidence_history) > 10:
                    st.session_state.confidence_history.pop(0)
                
                # Store last sources for export
                st.session_state.last_sources = sources
                
                # Phase 3A: Display response with enhanced template structure
                # Check if response follows Phase 3 template format
                template_sections = ai_response.get("template", {})
                if template_sections and len(template_sections) > 2:
                    # Render structured template
                    st.markdown("## " + template_sections.get("title", "Analysis Results"))
                    
                    if template_sections.get("executive_insight"):
                        st.markdown("### Executive Insight")
                        st.markdown(template_sections["executive_insight"])
                    
                    if template_sections.get("analysis"):
                        st.markdown("### Detailed Analysis")  
                        st.markdown(template_sections["analysis"])
                    
                    if template_sections.get("recommendations"):
                        st.markdown("### Recommendations")
                        st.markdown(template_sections["recommendations"])
                    
                    if template_sections.get("citations"):
                        st.markdown("### Citations")
                        st.markdown(template_sections["citations"])
                    
                    if template_sections.get("limits_data_needed"):
                        st.markdown("### Limits/Missing Data")
                        st.markdown(template_sections["limits_data_needed"])
                else:
                    # Fallback to regular markdown display
                    message_placeholder.markdown(answer)
                
                # Phase 3C: Enhanced confidence badge display
                confidence_data = ai_response.get("confidence_data", {})
                if confidence_data:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**Confidence:** {confidence_data.get('badge', 'Medium')} ({confidence_data.get('score', 0.5):.3f})")
                    with col2:
                        if confidence_data.get("escalation_recommended"):
                            st.warning(f"🔄 Escalated to {confidence_data['escalation_recommended']}")
                    with col3:
                        if confidence_data.get("breakdown"):
                            with st.expander("📊 R/A/V/C Breakdown"):
                                breakdown = confidence_data["breakdown"]
                                st.write(f"**R**etrieval: {breakdown.get('retrieval', 0):.2f}")
                                st.write(f"**A**nalysis: {breakdown.get('analysis', 0):.2f}")
                                st.write(f"**V**ariance: {breakdown.get('variance', 0):.2f}")
                                st.write(f"**C**overage: {breakdown.get('coverage', 0):.2f}")
                
                # Hidden debug caption (Intent=<auto_routed_intent> | Mode=<resolved_mode>)
                intent = ai_response.get("intent", "unknown")
                mode = "chat"  # This is chat mode
                st.caption(f"Intent={intent} | Mode={mode}")
                
                # Phase 3B: Enhanced sources display with coverage warnings
                sources_drawer = SourcesDrawer()
                coverage_warning = ai_response.get("kb_coverage_warning")
                if coverage_warning:
                    st.warning(f"⚠️ {coverage_warning}")
                sources_drawer.render_inline_sources(sources, confidence_score)
                
                # Add assistant message to chat history
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence_score,
                    "service_level": st.session_state.service_level
                })
                
            except Exception as e:
                error_message = f"❌ Error processing query: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "error": True
                })

# ─────────────────────────────────────────────────────────────────────────────
    # Enhanced Export Options
    # ─────────────────────────────────────────────────────────────
    if st.session_state.chat_messages:
        st.markdown("---")
        st.header("📤 Export Options")
        
        # Initialize export manager
        export_manager = ExportManager(st.session_state.service_level)
        
        # Prepare export data
        export_data = {
            "messages": st.session_state.chat_messages,
            "sources": st.session_state.last_sources,
            "confidence_history": st.session_state.confidence_history,
            "selected_files": st.session_state.selected_files,
            "metadata": {
                "conversation_id": st.session_state.current_conversation,
                "service_level": st.session_state.service_level,
                "exported_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_messages": len(st.session_state.chat_messages)
            }
        }
        
        # Add data gaps summary
        if "data_needed_panel" in st.session_state:
            gaps_summary = st.session_state.data_needed_panel.get_gaps_summary()
            export_data["data_gaps"] = gaps_summary
        
        # Export buttons in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📊 XLSX", use_container_width=True, help="Export to Excel with multiple sheets"):
                try:
                    xlsx_content = export_manager.export_to_xlsx(export_data)
                    st.download_button(
                        label="Download XLSX",
                        data=xlsx_content,
                        file_name=f"scie_ethos_export_{int(time.time())}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"XLSX export failed: {e}")
        
        with col2:
            if st.button("📄 Markdown", use_container_width=True, help="Export to Markdown format"):
                try:
                    md_content = export_manager.export_to_markdown(export_data)
                    st.download_button(
                        label="Download MD",
                        data=md_content,
                        file_name=f"scie_ethos_export_{int(time.time())}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Markdown export failed: {e}")
        
        with col3:
            if st.button("📝 DOCX", use_container_width=True, help="Export to Word document"):
                try:
                    docx_content = export_manager.export_to_docx(export_data)
                    st.download_button(
                        label="Download DOCX",
                        data=docx_content,
                        file_name=f"scie_ethos_export_{int(time.time())}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"DOCX export failed: {e}")
        
        with col4:
            if st.button("📊 PPTX", use_container_width=True, help="Export to PowerPoint"):
                try:
                    pptx_content = export_manager.export_to_pptx(export_data)
                    st.download_button(
                        label="Download PPTX",
                        data=pptx_content,
                        file_name=f"scie_ethos_export_{int(time.time())}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PPTX export failed: {e}")
        
        with col5:
            if st.button("🚀 All Formats", use_container_width=True, help="Export to all formats + Dropbox + S3"):
                with st.spinner("Exporting to all formats..."):
                    results = export_manager.export_all_formats(export_data)
                    
                    # Show results
                    st.success("Export completed!")
                    for format_name, success in results.items():
                        if success:
                            st.success(f"✅ {format_name.upper()}")
                        else:
                            st.error(f"❌ {format_name.upper()}")


# Main execution when run as standalone
if __name__ == "__main__":
    render_chat_assistant()
