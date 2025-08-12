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
from assistant_bridge import auto_model, run_query
from confidence import score_ravc, should_abstain
from path_utils import get_project_paths
from dbx_utils import list_data_files

# Set page configuration first
st.set_page_config(
    page_title="SCIE Ethos Analyst", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def get_confidence_badge(score: float) -> str:
    """Generate confidence badge with appropriate styling."""
    if score >= 0.8:
        css_class = "confidence-high"
        level = "HIGH"
    elif score >= 0.6:
        css_class = "confidence-medium"
        level = "MED"
    else:
        css_class = "confidence-low"
        level = "LOW"
    
    return f'<span class="badge {css_class}">{level} ({score:.2f})</span>'

def get_model_badge(intent: str = None) -> str:
    """Generate model badge based on current configuration."""
    if os.getenv("ASSISTANT_ID"):
        model_type = "assistant"
        display = "ASSISTANT"
        css_class = "model-assistant"
    else:
        # Use auto_model logic if available
        if intent and auto_model:
            model = auto_model(intent)
            if "4o" in model and "mini" not in model:
                model_type = "4o"
                display = "GPT-4o"
                css_class = "model-4o"
            else:
                model_type = "mini"
                display = "GPT-4o-mini"
                css_class = "model-mini"
        else:
            model_type = "pipeline"
            display = "PIPELINE"
            css_class = "model-mini"
    
    return f'<span class="badge {css_class}">{display}</span>'

def load_available_files() -> List[Dict[str, Any]]:
    """Load available cleansed files from Dropbox."""
    try:
        files = list_data_files(CLEANSED)
        return files
    except Exception as e:
        st.error(f"Could not load files from {CLEANSED}: {e}")
        return []

def format_conversation_name(name: str) -> str:
    """Format conversation name for display."""
    if name == "default":
        return "🏠 Main Conversation"
    return f"💬 {name}"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Conversation Management & System Status
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 SCIE Ethos Analyst")
    
    # ─── Conversation Management ───
    st.subheader("💬 Conversations")
    
    current_conv = st.session_state.current_conversation
    conv_display = format_conversation_name(current_conv)
    st.markdown(f"**Active:** {conv_display}")
    
    # Conversation rename/new
    with st.expander("Manage Conversations"):
        new_conv_name = st.text_input(
            "Conversation Name", 
            value=current_conv,
            key="conv_rename"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rename", use_container_width=True):
                st.session_state.current_conversation = new_conv_name
                st.rerun()
        
        with col2:
            if st.button("New Chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.current_conversation = f"chat_{int(time.time())}"
                st.session_state.last_sources = {}
                st.rerun()
    
    st.divider()
    
    # ─── System Status ───
    st.subheader("⚙️ System Status")
    
    # Model badge
    last_intent = st.session_state.last_sources.get("intent", "general")
    model_badge_html = get_model_badge(last_intent)
    st.markdown(f"**Model:** {model_badge_html}", unsafe_allow_html=True)
    
    # Confidence badge (from last interaction)
    if st.session_state.confidence_history:
        last_confidence = st.session_state.confidence_history[-1]
        confidence_badge_html = get_confidence_badge(last_confidence)
        st.markdown(f"**Confidence:** {confidence_badge_html}", unsafe_allow_html=True)
    
    st.divider()
    
    # ─── File Selection ───
    st.subheader("📊 Data Files")
    
    available_files = load_available_files()
    if available_files:
        file_options = []
        file_map = {}
        
        for file in available_files:
            display_name = f"{file['name']} ({file.get('file_type', 'excel')})"
            if file.get('server_modified'):
                display_name += f" — {file['server_modified'].strftime('%m/%d %H:%M')}"
            
            file_options.append(display_name)
            file_map[display_name] = file['path_lower']
        
        selected_displays = st.multiselect(
            "Select files for analysis:",
            options=file_options,
            default=st.session_state.selected_files,
            key="file_selector"
        )
        
        # Update session state
        st.session_state.selected_files = selected_displays
        selected_paths = [file_map[display] for display in selected_displays]
        
        if selected_paths:
            st.success(f"✅ {len(selected_paths)} file(s) selected")
        else:
            st.info("💡 Select files to focus analysis")
    else:
        st.warning("No cleansed files found")
    
    st.divider()
    
    # ─── Project Info ───
    st.subheader("🏗️ Project Paths")
    with st.expander("System Paths"):
        st.code(f"Project Root: {PROJECT_ROOT}")
        st.code(f"Data Root: {DATA_ROOT}")
        st.code(f"Metadata: {META_DIR}")
        st.code(f"Cleansed Files: {CLEANSED}")

# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────────────────────────────────────
st.title("💬 Chat")

# Display chat history
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display artifacts if present
        if message.get("artifacts"):
            with st.expander("📎 Artifacts"):
                for artifact in message["artifacts"]:
                    artifact_path = str(artifact)
                    
                    # Handle different artifact types
                    if artifact_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        if Path(artifact_path).exists():
                            st.image(artifact_path, use_container_width=True)
                        else:
                            st.caption(f"🖼️ Chart: {artifact_path}")
                    
                    elif artifact_path.lower().endswith((".xlsx", ".csv")):
                        st.caption(f"📊 Data: {artifact_path}")
                    
                    elif artifact_path.lower().endswith(".json"):
                        st.caption(f"📄 Metadata: {artifact_path}")
                    
                    else:
                        st.caption(f"📎 {artifact_path}")

# Chat input
user_question = st.chat_input("Ask about data, docs, KB, comparisons…")

if user_question:
    # Add user message to chat
    st.session_state.chat_messages.append({
        "role": "user", 
        "content": user_question
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Process assistant response
    with st.chat_message("assistant"):
        # Show thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤔 _Analyzing your question..._")
        
        start_time = time.time()
        
        try:
            # Set up app_paths (simplified for chat interface)
            class AppPaths:
                def __init__(self):
                    # Use Dropbox paths from constants
                    self.dbx_cleansed_folder = CLEANSED
                    self.dbx_metadata_folder = META_DIR
                    self.dbx_summaries_folder = f"{DATA_ROOT}/03_Summaries"
            
            app_paths = AppPaths()
            
            # Get selected file paths
            selected_paths = []
            if st.session_state.selected_files:
                file_map = {}
                for file in available_files:
                    display_name = f"{file['name']} ({file.get('file_type', 'excel')})"
                    if file.get('server_modified'):
                        display_name += f" — {file['server_modified'].strftime('%m/%d %H:%M')}"
                    file_map[display_name] = file['path_lower']
                
                selected_paths = [file_map[display] for display in st.session_state.selected_files if display in file_map]
            
            # Call orchestrator
            result = answer_question(
                user_question=user_question,
                app_paths=app_paths,
                cleansed_paths=selected_paths if selected_paths else None,
                answer_style="concise"
            )
            
            # Extract answer
            answer = result.get("final_text", "")
            
            if not answer:
                # Fallback answer extraction
                answer = (
                    result.get("result", {}).get("answer") or
                    result.get("result", {}).get("plan") or
                    result.get("result", {}).get("analysis") or
                    f"_{result.get('intent_info', {}).get('intent', 'task')}_ completed."
                )
            
            # Calculate confidence
            confidence_score = score_ravc(
                recency=0.8, 
                alignment=0.9, 
                variance=0.2, 
                coverage=0.8
            )["score"]
            
            # Check for abstention
            if should_abstain(confidence_score, threshold=0.6):
                answer = f"⚠️ **Low Confidence Warning**\n\n{answer}\n\n_Please consider refining your question or providing more context._"
            
            # Update thinking indicator with final answer
            thinking_placeholder.markdown(answer)
            
            # Store sources and artifacts
            artifacts = result.get("artifacts", [])
            intent_info = result.get("intent_info", {})
            
            st.session_state.last_sources = {
                "intent": intent_info.get("intent"),
                "confidence": confidence_score,
                "tool_calls": result.get("tool_calls", []),
                "artifacts": artifacts,
                "kb_citations": result.get("kb_citations", []),
                "response_time": time.time() - start_time
            }
            
            # Add to confidence history
            st.session_state.confidence_history.append(confidence_score)
            if len(st.session_state.confidence_history) > 10:
                st.session_state.confidence_history.pop(0)
            
            # Add assistant message to chat
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
                "artifacts": artifacts
            })
            
            # Log the interaction
            try:
                st.session_state.session_handler.add_entry(user_question, result)
                log_query_result(
                    user_question, 
                    result, 
                    save_path=f"{META_DIR}/chat_log.jsonl"
                )
                log_event(f"Chat Q: {user_question[:60]}... Intent: {intent_info.get('intent')} Duration: {time.time() - start_time:.2f}s")
            except Exception as log_error:
                st.caption(f"⚠️ Logging failed: {log_error}")
        
        except Exception as e:
            error_message = f"❌ **Error Processing Request**\n\n```\n{str(e)}\n```\n\nPlease try rephrasing your question or check the selected data files."
            thinking_placeholder.markdown(error_message)
            
            # Add error to chat history
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": error_message,
                "artifacts": []
            })

# ─────────────────────────────────────────────────────────────────────────────
# Sources & Context Drawer
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.last_sources:
    with st.expander("🔍 Sources & Analysis Details"):
        sources = st.session_state.last_sources
        
        # Analysis metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            intent = sources.get("intent", "unknown")
            st.markdown(f"**Intent:** `{intent}`")
        
        with col2:
            confidence = sources.get("confidence", 0)
            confidence_html = get_confidence_badge(confidence)
            st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
        
        with col3:
            response_time = sources.get("response_time", 0)
            st.markdown(f"**Response Time:** {response_time:.2f}s")
        
        # Tool calls
        tool_calls = sources.get("tool_calls", [])
        if tool_calls:
            st.markdown("**🔧 Tools Used:**")
            for i, call in enumerate(tool_calls, 1):
                tool_name = call.get("tool", "unknown")
                result_meta = call.get("result_meta", {})
                
                with st.container():
                    st.markdown(f"**{i}.** `{tool_name}`")
                    if result_meta:
                        st.json(result_meta)
        
        # Knowledge base citations
        kb_citations = sources.get("kb_citations", [])
        if kb_citations:
            st.markdown("**📚 Knowledge Base Sources:**")
            for citation in kb_citations:
                st.markdown(f"- {citation}")
        
        # Artifacts
        artifacts = sources.get("artifacts", [])
        if artifacts:
            st.markdown("**📎 Generated Artifacts:**")
            for artifact in artifacts:
                st.code(str(artifact))

# ─────────────────────────────────────────────────────────────────────────────
# Export Options
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chat_messages:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Markdown", use_container_width=True):
            # Generate markdown export
            markdown_content = []
            markdown_content.append(f"# SCIE Ethos Chat Export")
            markdown_content.append(f"**Conversation:** {st.session_state.current_conversation}")
            markdown_content.append(f"**Exported:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
            markdown_content.append("")
            
            for message in st.session_state.chat_messages:
                role = message["role"].title()
                content = message["content"]
                markdown_content.append(f"## {role}")
                markdown_content.append("")
                markdown_content.append(content)
                
                if message.get("artifacts"):
                    markdown_content.append("")
                    markdown_content.append("**Artifacts:**")
                    for artifact in message["artifacts"]:
                        markdown_content.append(f"- {artifact}")
                
                markdown_content.append("")
            
            markdown_export = "\n".join(markdown_content)
            
            st.download_button(
                label="Download Conversation.md",
                data=markdown_export,
                file_name=f"scie_ethos_chat_{st.session_state.current_conversation}_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 Export JSON", use_container_width=True):
            # Generate JSON export with full metadata
            export_data = {
                "conversation_id": st.session_state.current_conversation,
                "exported_at": time.time(),
                "exported_at_human": time.strftime('%Y-%m-%d %H:%M:%S'),
                "messages": st.session_state.chat_messages,
                "last_sources": st.session_state.last_sources,
                "confidence_history": st.session_state.confidence_history,
                "selected_files": st.session_state.selected_files,
                "system_info": {
                    "project_root": PROJECT_ROOT,
                    "data_root": DATA_ROOT,
                    "metadata_dir": META_DIR
                }
            }
            
            json_export = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="Download Chat Data.json",
                data=json_export,
                file_name=f"scie_ethos_chat_data_{st.session_state.current_conversation}_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col3:
        st.markdown("**PDF Export**")
        st.caption("Requires additional setup\n(wkhtmltopdf/reportlab)")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "🧠 SCIE Ethos LLM Assistant | Architecture-Compliant Chat Interface"
    "</div>", 
    unsafe_allow_html=True
)
