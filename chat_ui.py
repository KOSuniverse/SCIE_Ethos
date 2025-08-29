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

# Page config is handled by main.py to ensure dark theme consistency

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

def _classify_query_type(prompt):
    """Classify query type for specialized processing."""
    prompt_lower = prompt.lower()
    
    # Direct reports / org chart queries
    if any(term in prompt_lower for term in ['direct reports', 'reports to', 'org chart', 'team members', 'staff']):
        return 'direct_reports'
    
    # Process/workflow queries
    if any(term in prompt_lower for term in ['ltsa', 'process', 'workflow', 'how does', 'procedure']):
        return 'process_workflow'
    
    # Inventory/systems queries
    if any(term in prompt_lower for term in ['inventory', 'system', 'erp', 'policy', 'provision']):
        return 'inventory_systems'
    
    # Meeting/participant queries
    if any(term in prompt_lower for term in ['meeting', 'participants', 'attendees', 'discussed']):
        return 'meeting_details'
    
    # Specific detail queries
    if any(term in prompt_lower for term in ['who', 'what', 'when', 'where', 'how much', 'details']):
        return 'specific_details'
    
    return 'general_query'

def _has_sufficient_detail(structured_data, query_type):
    """Check if structured data has sufficient detail for the query type."""
    if query_type == 'direct_reports':
        return 'direct_reports' in structured_data and len(structured_data.get('direct_reports', [])) > 0
    
    if query_type == 'process_workflow':
        return any(field in structured_data for field in ['ito_otr_flow', 'process', 'governance', 'operations_controls'])
    
    if query_type == 'inventory_systems':
        return any(field in structured_data for field in ['systems', 'policies', 'erps', 'entities'])
    
    if query_type == 'meeting_details':
        return 'participants' in structured_data and len(structured_data.get('participants', [])) > 0
    
    return True  # Assume sufficient for other types

def _create_structured_ai_prompt(prompt, query_type, structured_documents, drill_down_results):
    """Create specialized AI prompt based on query type and structured data."""
    
    # Prepare structured data for AI
    json_data_for_ai = []
    for doc in structured_documents:
        json_data_for_ai.append({
            "file_name": doc["file_name"],
            "folder": doc["folder"],
            "data": doc["structured_data"]
        })
    
    # Add drill-down results if available
    drill_down_content = ""
    if drill_down_results:
        drill_down_content = "\n\nADDITIONAL DRILL-DOWN ANALYSIS:\n"
        for result in drill_down_results:
            drill_down_content += f"\nFrom {result['source_file']}:\n{result['drill_down_content']}\n"
    
    if query_type == 'direct_reports':
        return f"""You are ChatGPT analyzing EthosEnergy organizational data. Answer this question: {prompt}

STRUCTURED DATA FROM DOCUMENTS:
{json.dumps(json_data_for_ai, indent=2)}

{drill_down_content}

INSTRUCTIONS:
- Extract and list ALL direct reports with their exact names and titles
- Use the "direct_reports" field from the JSON data
- Present in a natural, readable format like ChatGPT would
- Include the leader's name and title at the top
- If multiple documents show the same org structure, consolidate the information
- Be specific and complete - don't summarize away names or titles

Answer the question directly and comprehensively using the structured data provided."""

    elif query_type == 'process_workflow':
        return f"""You are ChatGPT analyzing EthosEnergy business processes. Answer this question: {prompt}

STRUCTURED DATA FROM DOCUMENTS:
{json.dumps(json_data_for_ai, indent=2)}

{drill_down_content}

INSTRUCTIONS:
- Extract process flows, workflows, and operational details from the JSON data
- Look for fields like "ito_otr_flow", "contract_economics", "operations_controls", "process", "governance"
- Present information in clear sections (flow, economics, controls, risks, actions)
- Include specific details like timelines, responsibilities, and procedures
- Be comprehensive but well-organized like ChatGPT would structure it

Answer naturally using all the structured process information provided."""

    elif query_type == 'inventory_systems':
        return f"""You are ChatGPT analyzing EthosEnergy inventory and systems data. Answer this question: {prompt}

STRUCTURED DATA FROM DOCUMENTS:
{json.dumps(json_data_for_ai, indent=2)}

{drill_down_content}

INSTRUCTIONS:
- Extract inventory policies, systems, entities, and operational details
- Look for "systems", "entities", "policies", "erps", "scope", "challenges" in the JSON data
- Organize by scope, policies, systems, challenges, and actions
- Include specific system names, entities, and policy details
- Present in a structured but natural ChatGPT-style response

Use all the structured inventory and systems data to provide a comprehensive answer."""

    else:
        # General structured response
        return f"""You are ChatGPT analyzing EthosEnergy business documents. Answer this question: {prompt}

STRUCTURED DATA FROM DOCUMENTS:
{json.dumps(json_data_for_ai, indent=2)}

{drill_down_content}

INSTRUCTIONS:
- Use ALL the structured data provided in the JSON format
- Extract specific names, roles, processes, amounts, dates, and details
- Answer naturally and comprehensively like ChatGPT would
- Focus on what the user actually asked for
- Include relevant details from participants, actions, deliverables, and other structured fields
- If information is missing, state that clearly

Provide a thorough, natural response using the structured business data provided."""

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
    
    # Enhanced dark theme styling with stronger dark preferences
    st.markdown("""
    <style>
        /* Force dark theme elements */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Chat messages styling - enhanced for dark theme */
        .stChatMessage {
            font-size: 1rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            border: 1px solid #262730;
            background-color: #1E1E1E;
        }
        
        /* Enhanced badges with better dark theme contrast */
        .badge {
            padding: 4px 8px;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin: 2px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .confidence-high { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }
        .confidence-medium { background: linear-gradient(135deg, #FF9800 0%, #f57c00 100%); }
        .confidence-low { background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%); }
        .model-mini { background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); }
        .model-4o { background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%); }
        .model-assistant { background: linear-gradient(135deg, #00BCD4 0%, #0097A7 100%); }
        
        /* Enhanced sources card for dark theme */
        .sources-card {
            background: #1E1E1E;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
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
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Enhanced button styling for dark theme */
        .stButton button {
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
            background-color: #262730;
            border: 1px solid #444;
            color: #FAFAFA;
        }
        
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
            background-color: #4CAF50;
            border-color: #4CAF50;
        }
        
        /* Chat input styling for dark theme */
        .stChatInput input {
            border-radius: 20px;
            background-color: #262730;
            border: 1px solid #444;
            color: #FAFAFA;
        }
        
        /* Sidebar enhancements for dark theme */
        .css-1d391kg {
            background-color: #0E1117;
        }
        
        /* Enhanced text areas and inputs */
        .stTextInput input, .stTextArea textarea {
            background-color: #262730;
            border: 1px solid #444;
            color: #FAFAFA;
            border-radius: 8px;
        }
        
        /* Dark theme tip box */
        .dark-theme-tip {
            background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
            border: 1px solid #4CAF50;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            color: #FAFAFA;
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
                                
                                # Collect all structured JSON data for AI analysis
                                structured_documents = []
                                
                                for result in indexed_results:
                                    file_name = result.get('file_name', 'Unknown')
                                    folder = result.get('folder', 'root')
                                    summary = result.get('summary', 'No summary available')
                                    relevance = result.get('relevance_score', 0)
                                    original_path = result.get('file_path', '')
                                    original_json = result.get('original_json', {})
                                    
                                    # Clean up any OpenAI citation formats in the summary
                                    import re
                                    summary = re.sub(r'【\d+:\d+†[^】]*】', '', summary)
                                    summary = re.sub(r'【[^】]*】', '', summary)
                                    
                                    # Store structured data for AI processing
                                    structured_documents.append({
                                        "file_name": file_name,
                                        "folder": folder,
                                        "original_path": original_path,
                                        "relevance": relevance,
                                        "structured_data": original_json,
                                        "summary": summary
                                    })
                                    
                                    # Enhanced display with drill-down capability
                                    kb_answer += f"**📄 From \"{file_name}\"**\n"
                                    kb_answer += f"*Location: {folder}*\n"
                                    if original_path:
                                        kb_answer += f"*Raw Document: `{original_path}`*\n"
                                    kb_answer += f"*Relevance: {relevance:.1%}*\n\n"
                                    
                                    # Show structured data preview if available
                                    if original_json:
                                        if 'direct_reports' in original_json:
                                            kb_answer += f"**👥 DIRECT REPORTS FOUND:** {len(original_json['direct_reports'])} people\n"
                                        if 'participants' in original_json:
                                            kb_answer += f"**👥 PARTICIPANTS:** {', '.join(original_json['participants'][:3])}{'...' if len(original_json['participants']) > 3 else ''}\n"
                                        if 'actions' in original_json:
                                            kb_answer += f"**✅ ACTIONS:** {len(original_json['actions'])} action items\n"
                                    
                                    kb_answer += f"**📋 SUMMARY:**\n{summary}\n\n"
                                    kb_answer += f"💡 *Drill-down available to raw document*\n\n"
                                    kb_answer += "---\n\n"
                                    
                                    kb_sources.append({
                                        "name": file_name,
                                        "folder": folder,
                                        "relevance": relevance,
                                        "original_path": original_path,
                                        "drill_down_available": bool(original_path),
                                        "structured_data": original_json
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
                    # KB summaries found - use structured JSON data for AI analysis
                    
                    # Determine query type for specialized prompting
                    query_type = _classify_query_type(prompt)
                    
                    # Check if drill-down is needed
                    drill_down_results = []
                    if query_type in ['direct_reports', 'specific_details'] and structured_documents:
                        for doc in structured_documents[:2]:  # Drill down top 2 most relevant
                            if doc['original_path'] and not _has_sufficient_detail(doc['structured_data'], query_type):
                                print(f"🔍 Drilling down to {doc['file_name']} for more detail...")
                                drill_result = indexer.drill_down_to_raw_document(doc['original_path'], prompt)
                                if drill_result:
                                    drill_down_results.append(drill_result)
                    
                    # Create structured prompt with JSON data
                    enhanced_ai_prompt = _create_structured_ai_prompt(
                        prompt, query_type, structured_documents, drill_down_results
                    )
                    
                    try:
                        # Use Chat Completion API directly to avoid Assistant's attached training docs
                        from openai import OpenAI
                        client = OpenAI()
                        
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are ChatGPT, a helpful AI assistant. Answer questions naturally and conversationally based on the provided information. Include specific details like names, roles, and departments when they exist in the documents. Be thorough but natural - don't use rigid templates."},
                                {"role": "user", "content": enhanced_ai_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=600
                        )
                        
                        ai_answer = response.choices[0].message.content
                        
                        # Clean up any citations that might slip through
                        import re
                        ai_answer = re.sub(r'【\d+:\d+†[^】]*】', '', ai_answer)
                        ai_answer = re.sub(r'【[^】]*】', '', ai_answer)
                        
                    except Exception as e:
                        ai_answer = "The meeting documents above provide the specific information available."
                else:
                    # No KB findings - provide ChatGPT-like general knowledge response
                    general_prompt = f"""You are a knowledgeable business assistant. Please provide a comprehensive, helpful response to: {prompt}

RESPONSE GUIDELINES:
- Provide detailed, useful information like ChatGPT would
- Use your general knowledge and expertise to give practical insights
- Include relevant business concepts, frameworks, or best practices when applicable
- Make the response actionable and valuable
- Structure information clearly with headings or bullet points when helpful
- Do NOT cite specific training documents, PDFs, or external company examples
- Do NOT use citation formats like 【4:0†source】
- Focus on providing genuine value and practical guidance
- If the question is about specific company data you don't have, clearly state that and pivot to general guidance

Remember: You should be as helpful and comprehensive as ChatGPT, while avoiding citations to specific documents."""
                    
                    try:
                        # Use Chat Completion API for comprehensive general knowledge
                        from openai import OpenAI
                        client = OpenAI()
                        
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are an expert business consultant and analyst. Provide comprehensive, practical, and actionable advice using your extensive knowledge. Be as helpful as ChatGPT while avoiding citations to specific documents."},
                                {"role": "user", "content": general_prompt}
                            ],
                            temperature=0.4,
                            max_tokens=800
                        )
                        
                        ai_answer = response.choices[0].message.content
                        
                        # Clean up any citations
                        import re
                        ai_answer = re.sub(r'【\d+:\d+†[^】]*】', '', ai_answer)
                        ai_answer = re.sub(r'【[^】]*】', '', ai_answer)
                    except:
                        ai_answer = "I don't have specific information to answer this question without access to relevant documents."
                
                # Create dual format answer with clear labeling
                if kb_answer and "No Relevant Meeting Documents Found" not in kb_answer:
                    answer = f"""### 📚 Document-Based Answer
*Based on your EthosEnergy meeting summaries and documents*

{kb_answer}

### 🤖 AI Analysis
*Synthesized insights from your documents*

{ai_answer}"""
                else:
                    answer = f"""### 🧠 General Knowledge Response
*This answer uses general business knowledge as no relevant documents were found*

{ai_answer}

💡 **Note**: This response is based on general knowledge, not your specific EthosEnergy documents. To get document-based answers, try uploading relevant files or ensure your question matches content in your knowledge base."""
                
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
