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

def _analyze_query_type(prompt):
    """Analyze the query to determine the best response format."""
    prompt_lower = prompt.lower()
    
    # List queries
    if any(word in prompt_lower for word in ['list', 'show all', 'what are', 'give me all']):
        if any(word in prompt_lower for word in ['part', 'product', 'sku', 'number']):
            return 'part_list'
        elif any(word in prompt_lower for word in ['meeting', 'quarterly', 'siop']):
            return 'meeting_list'
        else:
            return 'general_list'
    
    # Data/inventory queries
    if any(word in prompt_lower for word in ['inventory', 'stock', 'level', 'wip', 'data']):
        return 'data_query'
    
    # Meeting summaries
    if any(word in prompt_lower for word in ['summarize', 'summary', 'what happened', 'discuss']):
        return 'summary_query'
    
    # Specific information requests
    if any(word in prompt_lower for word in ['who', 'when', 'where', 'how much', 'what decision']):
        return 'specific_query'
    
    return 'general_query'

def _create_adaptive_analysis_prompt(prompt, query_type, uploaded_files, file_list, conversation_context):
    """Create an adaptive prompt based on the query type."""
    
    base_info = f"""I have uploaded {len(uploaded_files)} documents to analyze. Please answer this question: {prompt}

UPLOADED DOCUMENTS:
{file_list}{conversation_context}

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:
- Use ONLY this source format: "From [DOCUMENT NAME] (folder: [FOLDER]): [specific detail]"
- NEVER use 【4:1†source】 or similar OpenAI citation formats - I will reject responses with this format
- Extract SPECIFIC, ACTIONABLE details - not high-level summaries
- Focus on information that can be used to solve problems or answer follow-up questions
- NEVER include textbook examples, academic exercises, or generic company data (like "Pegasa" or random manufacturers)
- Use ONLY actual EthosEnergy business data and real operational information from the uploaded documents
- If you don't find specific information in the documents, say so clearly - don't make up examples

"""
    
    if query_type == 'part_list':
        return base_info + """
RESPONSE FORMAT FOR PART/PRODUCT LIST:
Provide a simple, clean list format:

**Part Numbers Found:**
• [Part Number] - [Product Name] (from [Document])
• [Part Number] - [Product Name] (from [Document])

Include ONLY actual part numbers mentioned in the documents. Do not include textbook examples or generic inventory data.
If no specific part numbers are found, state: "No specific part numbers found in the uploaded documents."
"""
    
    elif query_type == 'data_query':
        return base_info + """
RESPONSE FORMAT FOR DATA/INVENTORY QUERY:
Focus on ACTUAL operational data from your company:

**Current Data Found:**
• [Specific metric]: [Actual value] (from [Document])
• [Inventory item]: [Current level] (from [Document])

EXCLUDE textbook examples, academic exercises, or generic inventory management data.
Include ONLY real operational data from EthosEnergy or related business operations.
"""
    
    elif query_type == 'summary_query':
        return base_info + """
RESPONSE FORMAT FOR MEETING SUMMARY:
Provide a structured but natural summary:

**Meeting: [Meeting Name/Date]**
**Key Participants:** [Names and roles]
**Main Decisions:**
• [Decision 1] - Responsible: [Person]
• [Decision 2] - Responsible: [Person]

**Action Items:**
• [Action] - Due: [Date] - Owner: [Person]

**Key Discussion Points:**
• [Specific topic with details]

Focus on ACTIONABLE information that could be referenced later.
"""
    
    elif query_type == 'specific_query':
        return base_info + """
RESPONSE FORMAT FOR SPECIFIC INFORMATION:
Answer the specific question directly and concisely:

[Direct answer to the question]

**Supporting Details:**
• [Specific detail 1] (from [Document])
• [Specific detail 2] (from [Document])

Keep the response focused on exactly what was asked.
"""
    
    else:  # general_query
        return base_info + """
RESPONSE FORMAT:
Provide a natural, comprehensive answer organized logically:

[Main answer to the question]

**Key Details:**
• [Important detail 1] (from [Document])
• [Important detail 2] (from [Document])

**Additional Context:**
[Any relevant background information]

Organize information in a way that best serves the specific question asked.
"""

def render_chat_assistant():
    """Render the complete chat assistant UI inline."""
    
    # Enhanced styling that works with both light and dark themes
    st.markdown("""
    <style>
        /* Chat messages styling - adapts to theme */
        .stChatMessage {
            font-size: 1rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            border: 1px solid var(--secondary-background-color);
        }
        
        /* Badges that work in both themes */
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
        
        /* Sources card that adapts to theme */
        .sources-card {
            background: var(--secondary-background-color);
            border: 1px solid var(--border-color);
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
        
        /* Enhanced button styling */
        .stButton button {
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Chat input styling */
        .stChatInput input {
            border-radius: 20px;
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
        
        # Theme toggle info
        st.info("💡 **Tip**: Toggle dark/light mode using the ⚙️ settings menu (top-right)")
        
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
        
        # File Upload for Document Search
        st.subheader("📎 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload documents for analysis",
            type=['pdf', 'docx', 'pptx', 'txt', 'md', 'csv', 'xlsx'],
            accept_multiple_files=True,
            help="Upload documents to get specific answers from your files"
        )
        
        if uploaded_files:
            st.session_state.selected_files = uploaded_files
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            for file in uploaded_files:
                st.write(f"📄 {file.name}")
        else:
            st.session_state.selected_files = []
        
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
        
        # KB Indexer Section
        st.subheader("📚 Knowledge Base Tools")
        
        if st.button("🔄 Index New Documents", help="Process new documents and create searchable summaries"):
            with st.spinner("Indexing new documents..."):
                try:
                    import sys
                    import os
                    sys.path.append('PY Files')
                    from kb_indexer import KBIndexer
                    
                    kb_path = os.getenv('KB_DBX_PATH', '/Project_Root/06_LLM_Knowledge_Base')
                    data_path = os.getenv('DATA_DBX_PATH', '/Project_Root/04_Data')
                    
                    indexer = KBIndexer(kb_path, data_path)
                    result = indexer.process_new_files()
                    
                    st.success(f"✅ Indexing Complete!")
                    st.info(f"""📊 **Results:**
- Processed: {result.get('processed', 0)} new files
- Failed: {result.get('failed', 0)} files  
- Skipped (unchanged): {result.get('skipped', 0)} files
- Skipped (in FAISS): {result.get('faiss_skipped', 0)} files
- Total index size: {result.get('index_size', 0)} documents""")
                    
                except Exception as e:
                    st.error(f"❌ Indexing failed: {str(e)}")
        
        st.caption("Creates searchable summaries for new documents while respecting your existing FAISS index.")

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
                
                                # DOCUMENT SEARCH - Search through uploaded documents (not KB folder)
                kb_answer = "📋 **Searching uploaded documents...**"
                kb_sources = []
                
                try:
                    if st.session_state.selected_files:
                        # UPLOADED FILES: Search through user-uploaded documents
                        print(f"🔍 Searching through {len(st.session_state.selected_files)} uploaded files")
                        
                        from openai import OpenAI
                        import os
                        client = OpenAI()
                        
                        def _get_assistant_id():
                            return os.getenv("ASSISTANT_ID", "asst_abc123")
                        
                        doc_search_prompt = f"""Based on the documents I've uploaded to this conversation, please provide a comprehensive answer to this question: {prompt}

INSTRUCTIONS:
- Use ONLY information from the uploaded documents
- Provide specific details, names, dates, and numbers from the actual documents
- Cite which document each piece of information comes from
- If the documents don't contain relevant information, clearly state this
- Focus on actionable, specific information that can be used for decision-making

Question: {prompt}

Please structure your response with clear sections and cite your sources."""
                        
                        if st.session_state.thread_id:
                            thread = client.beta.threads.retrieve(st.session_state.thread_id)
                            
                            client.beta.threads.messages.create(
                                thread_id=thread.id,
                                role="user",
                                content=doc_search_prompt
                            )
                            
                            run = client.beta.threads.runs.create_and_poll(
                                thread_id=thread.id,
                                assistant_id=_get_assistant_id(),
                                temperature=0.2,
                                max_completion_tokens=1500
                            )
                            
                            if run.status == 'completed':
                                messages = client.beta.threads.messages.list(thread_id=thread.id)
                                kb_answer = messages.data[0].content[0].text.value
                                
                                import re
                                kb_answer = re.sub(r'【\d+:\d+†[^】]*】', '', kb_answer)
                                kb_answer = re.sub(r'【[^】]*】', '', kb_answer)
                                
                                kb_sources = [{"name": f"Uploaded file {i+1}", "type": "document"} for i in range(len(st.session_state.selected_files))]
                                print(f"✅ Document search complete: {len(kb_answer)} characters")
                            else:
                                kb_answer = f"📋 **Document Search Failed**\n\nCould not analyze uploaded documents (status: {run.status})"
                        else:
                            kb_answer = f"📋 **No Active Thread**\n\nPlease ask a question first to establish a conversation thread."
                    
                    else:
                        # KB SUMMARIES: Search through your indexed meeting summaries (not FAISS training docs)
                        print("🔍 Searching KB summaries from meeting minutes and emails")
                        
                        try:
                            import sys
                            sys.path.append('PY Files')
                            from kb_indexer import KBIndexer
                            import os
                            
                            kb_path = os.getenv('KB_DBX_PATH', '/Project_Root/06_LLM_Knowledge_Base')
                            data_path = os.getenv('DATA_DBX_PATH', '/Project_Root/04_Data')
                            
                            indexer = KBIndexer(kb_path, data_path)
                            indexed_results = indexer.search_summaries(prompt, max_results=5)
                            
                            if indexed_results:
                                kb_answer = f"Found {len(indexed_results)} relevant meeting documents:\n\n"
                                kb_sources = []
                                
                                for result in indexed_results:
                                    file_name = result.get('file_name', 'Unknown')
                                    folder = result.get('folder', 'root')
                                    summary = result.get('summary', 'No summary available')
                                    relevance = result.get('relevance_score', 0)
                                    
                                    # Clean up any OpenAI citation formats in the summary
                                    import re
                                    summary = re.sub(r'【\d+:\d+†[^】]*】', '', summary)
                                    summary = re.sub(r'【[^】]*】', '', summary)
                                    
                                    kb_answer += f"**From \"{file_name}\" (folder: {folder}):**\n"
                                    kb_answer += f"{summary}\n\n"
                                    
                                    kb_sources.append({
                                        "name": file_name,
                                        "folder": folder,
                                        "relevance": relevance
                                    })
                                
                                print(f"✅ Found {len(indexed_results)} relevant meeting summaries")
                            else:
                                kb_answer = f"📋 **No Relevant Meeting Documents Found**\n\nNo meeting minutes or emails in the knowledge base match your query: '{prompt}'"
                                print("📋 No relevant summaries found in KB_Summaries")
                                
                        except Exception as kb_error:
                            kb_answer = f"📋 **KB Search Error**\n\nCould not search meeting summaries: {str(kb_error)}\n\nTip: Upload documents using the file uploader for document-based answers."
                            print(f"❌ KB summary search failed: {kb_error}")
                        
                except Exception as e:
                    kb_answer = f"📋 **Document Search Error**\n\nCould not search uploaded documents: {str(e)}"
                    print(f"❌ Document search failed: {e}")
                
                # Combine responses with enhanced AI analysis
                ai_answer = ai_response.get("answer", "No AI response available")
                
                # Enhance AI analysis based on whether we have KB findings or not
                if st.session_state.selected_files:
                    # Files uploaded - AI can use the thread context
                    ai_answer = ai_response.get("answer", "No AI response available")
                elif kb_answer and "No Relevant Meeting Documents Found" not in kb_answer and "KB Search Error" not in kb_answer:
                    # KB summaries found - provide contextual analysis
                    enhanced_ai_prompt = f"""Based on the meeting summaries and documents from EthosEnergy below, provide additional context, analysis, or insights that relate specifically to the question: {prompt}

Meeting Documents Summary:
{kb_answer}

CRITICAL INSTRUCTIONS:
- Build upon the meeting findings, don't repeat them
- Add relevant context or implications ONLY from your general business knowledge about S&OP, supply chain, and operations
- NEVER reference external companies, their data, or training materials
- NEVER cite documents like "Eckert_Lauren_S_OPDesignforaHighEndJewelryRetailer.pdf" or other training PDFs
- Focus on general principles, best practices, or analytical frameworks
- Do NOT use citations with 【4:0†source】 format
- Relate your analysis directly to EthosEnergy's specific situation from the meeting documents
- Focus on insights that would help with follow-up questions or decisions about EthosEnergy operations"""
                    
                    try:
                        # Get contextual AI response without file attachments to avoid FAISS training docs
                        ai_response_enhanced = run_query(enhanced_ai_prompt)
                        ai_answer = ai_response_enhanced.get("answer", ai_answer)
                        
                        # Clean up any training document citations that might slip through
                        import re
                        ai_answer = re.sub(r'【\d+:\d+†[^】]*】', '', ai_answer)
                        ai_answer = re.sub(r'【[^】]*】', '', ai_answer)
                        
                    except:
                        ai_answer = "The meeting documents above provide the specific information available."
                else:
                    # No KB findings - provide general response without training docs
                    general_prompt = f"""Provide a brief, general response to this question: {prompt}

CRITICAL INSTRUCTIONS:
- Do NOT cite any training documents, PDFs, or external company examples
- Do NOT use citations with 【4:0†source】 format
- Provide only general business knowledge without specific company references
- Keep response concise and focused on the question asked
- If you don't have relevant general knowledge, state: "I don't have specific information to answer this question without access to relevant documents."
"""
                    
                    try:
                        ai_response_enhanced = run_query(general_prompt)
                        ai_answer = ai_response_enhanced.get("answer", "I don't have specific information to answer this question without access to relevant documents.")
                        
                        # Clean up any citations
                        import re
                        ai_answer = re.sub(r'【\d+:\d+†[^】]*】', '', ai_answer)
                        ai_answer = re.sub(r'【[^】]*】', '', ai_answer)
                    except:
                        ai_answer = "I don't have specific information to answer this question without access to relevant documents."
                
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
