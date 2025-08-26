#!/usr/bin/env python3
"""
KB Document Indexer - Automatically summarize and index knowledge base documents
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Import existing utilities
try:
    from dropbox_kb_sync import list_kb_candidates
    from dbx_utils import read_file_bytes, upload_dropbox_file_to_openai
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False
    print("Warning: Dropbox utilities not available")

class KBIndexer:
    def __init__(self, kb_path: str, data_path: str, summaries_path: str = "/KB_Summaries"):
        self.kb_path = kb_path
        self.data_path = data_path
        self.summaries_path = summaries_path
        self.client = OpenAI()
        
        # Dropbox paths for saving summaries
        self.dropbox_summaries_folder = "/Project_Root/06_LLM_Knowledge_Base/KB_Summaries"
        self.master_index_path = "/Project_Root/06_LLM_Knowledge_Base/KB_Summaries/master_index.json"
        
        # Load assistant configuration
        self.assistant_id = self._get_assistant_id()
        
        # Load existing indexes
        self.document_index = self._load_document_index()
        self.file_hashes = self._load_file_hashes()
        self.faiss_indexed_files = self._load_faiss_indexed_files()
    
    def _get_assistant_id(self) -> Optional[str]:
        """Get OpenAI Assistant ID from configuration."""
        try:
            with open("prompts/assistant.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
                return meta["assistant_id"]
        except:
            return os.getenv("ASSISTANT_ID")
    
    def _load_document_index(self) -> Dict[str, Any]:
        """Load existing document summaries index."""
        index_path = f"{self.summaries_path}/document_summaries.json"
        try:
            if DROPBOX_AVAILABLE:
                # Try to read from Dropbox
                from dbx_utils import dbx_read_json
                return dbx_read_json(index_path) or {}
        except:
            pass
        return {}
    
    def _load_file_hashes(self) -> Dict[str, str]:
        """Load file hashes to track processed files."""
        hash_path = f"{self.summaries_path}/file_hashes.json"
        try:
            if DROPBOX_AVAILABLE:
                from dbx_utils import dbx_read_json
                return dbx_read_json(hash_path) or {}
        except:
            pass
        return {}
    
    def _load_faiss_indexed_files(self) -> set:
        """Load list of files already in FAISS index to avoid reprocessing."""
        try:
            if DROPBOX_AVAILABLE:
                # Try to read manifest.json or docstore.pkl to get indexed files
                manifest_path = f"{self.kb_path}/manifest.json"
                docstore_path = f"{self.kb_path}/docstore.pkl"
                
                indexed_files = set()
                
                # Try manifest first
                try:
                    from dbx_utils import dbx_read_json
                    manifest = dbx_read_json(manifest_path)
                    if manifest and isinstance(manifest, dict):
                        # Extract file paths from manifest
                        for key, value in manifest.items():
                            if isinstance(value, dict) and 'source' in value:
                                indexed_files.add(value['source'])
                            elif isinstance(value, str) and value.startswith('/'):
                                indexed_files.add(value)
                except:
                    pass
                
                # If no manifest, try to infer from docstore
                if not indexed_files:
                    try:
                        # This is a simplified approach - in reality we'd need to parse the pickle file
                        # For now, we'll be conservative and assume FAISS has processed files
                        # User can override by clearing the FAISS index if needed
                        print("📋 FAISS index detected - will skip files that appear to be already indexed")
                    except:
                        pass
                
                return indexed_files
                
        except Exception as e:
            print(f"Warning: Could not load FAISS index info: {e}")
        
        return set()
    
    def _save_document_index(self):
        """Save document summaries index."""
        index_path = f"{self.summaries_path}/document_summaries.json"
        try:
            if DROPBOX_AVAILABLE:
                from dbx_utils import upload_json
                upload_json(index_path, self.document_index)
        except Exception as e:
            print(f"Failed to save document index: {e}")
    
    def _save_file_hashes(self):
        """Save file hashes."""
        hash_path = f"{self.summaries_path}/file_hashes.json"
        try:
            if DROPBOX_AVAILABLE:
                from dbx_utils import upload_json
                upload_json(hash_path, self.file_hashes)
        except Exception as e:
            print(f"Failed to save file hashes: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for change detection."""
        try:
            if DROPBOX_AVAILABLE:
                content = read_file_bytes(file_path)
                return hashlib.md5(content).hexdigest()
        except:
            pass
        return str(hash(file_path))  # Fallback
    
    def _create_summary_prompt(self, file_name: str, file_type: str, folder_path: str) -> str:
        """Create appropriate summarization prompt based on file type."""
        
        base_instructions = f"""
CRITICAL: Create a GRANULAR, DATA-MINING level summary for this document.

This summary will be used to FIND this document later when searching for specific information. Include ALL searchable details that someone might look for.

Extract and include EVERY:
- Person name (full name, title, role, department) - e.g., "John Smith, VP Operations", "Sarah Johnson, Finance Manager"  
- Product name, model number, SKU, part number - e.g., "Widget-A Model 2024", "SKU-12345", "Product Line B"
- Company name, vendor, supplier, customer - e.g., "ABC Manufacturing", "XYZ Logistics"
- Location, facility, region, country - e.g., "Ohio Plant", "European Division", "Site #3"
- Date mentioned (meeting dates, deadlines, launch dates) - e.g., "March 15, 2024", "Q2 deadline"
- Number, quantity, percentage, dollar amount - e.g., "$2.3M budget", "15% increase", "50,000 units"
- Decision made and decision maker - e.g., "John Smith approved 20% price increase"
- Action item and responsible party - e.g., "Sarah to review contracts by April 30"
- Problem identified and impact - e.g., "12,000 unit shortage causing $450K revenue loss"
- Solution proposed and timeline - e.g., "Expedite supplier delivery, 2-week timeline"
- Key topic, theme, discussion point - Include specific context, not just categories

SEARCHABILITY FOCUS: Include terms someone would actually search for:
- If discussing inventory: Include specific product names, quantities, locations
- If discussing budgets: Include dollar amounts, cost centers, approval levels  
- If discussing people: Include full names, titles, what they said/decided
- If discussing timelines: Include specific dates, milestones, deadlines
- If discussing problems: Include specific impacts, root causes, solutions

Format as a comprehensive summary that reads like: "On [DATE], [PERSON] discussed [SPECIFIC TOPIC] involving [SPECIFIC DETAILS]..."

Document: {file_name}
Location: {folder_path}
"""
        
        if file_type in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            return f"""{base_instructions}

This is an IMAGE file. Please provide:

**METADATA SECTION:**
- Image Type: [Photo, diagram, chart, screenshot, etc.]
- People Identified: [Names and roles of people in the image]
- Date Visible: [Any dates visible in the image or metadata]
- Location/Context: [Where was this taken or what does it show?]

**CONTENT SECTION:**
- Detailed description of what's shown in the image
- All text visible in the image (signs, labels, documents, etc.)
- Names and titles of people if this is a team/leadership photo
- Organizations, departments, or groups represented
- Any charts, graphs, or data visualized
- Meeting rooms, locations, or settings visible

**SEARCHABLE SUMMARY:**
Create a comprehensive description that includes all visible text, people names, and context.
Focus on making this searchable for finding specific people or information.

Format as clear, searchable text with all visible details captured."""

        elif file_type in ['.msg', '.eml']:
            return f"""{base_instructions}

This is an EMAIL file. Extract EVERY searchable detail:

**METADATA SECTION:**
- Email Date: [Extract exact date/time]
- From: [Full sender name, title if mentioned, email address]
- To: [All recipient names and emails]
- Subject: [Complete subject line]
- CC/BCC: [All additional recipients]

**GRANULAR CONTENT EXTRACTION:**
- Full names and titles of everyone mentioned
- Specific product names, model numbers, SKUs discussed
- Exact dollar amounts, quantities, percentages mentioned
- Specific dates (deadlines, meeting dates, launch dates)
- Decisions made and who made them
- Action items with responsible parties and deadlines
- Problems identified with specific impacts
- Solutions proposed with timelines
- Company names, vendors, suppliers mentioned
- Locations, facilities, regions discussed
- Project names, initiative codes, reference numbers

**EMAIL THREAD CONTEXT:**
- If this is part of a thread, include context from previous messages
- Reference any attachments mentioned
- Note any urgency indicators or priority levels

**SEARCHABLE SUMMARY:**
Create a comprehensive summary: "On [EXACT DATE], [FULL NAME/TITLE] emailed [RECIPIENTS] regarding [SPECIFIC TOPIC]. Key points: [SPECIFIC PERSON] decided [SPECIFIC DECISION] involving [SPECIFIC PRODUCTS/AMOUNTS]. Action items: [PERSON] to [SPECIFIC ACTION] by [DATE]. Problems discussed: [SPECIFIC ISSUES] with [DOLLAR IMPACT]. Solutions: [SPECIFIC PROPOSALS]."

Include ALL names, numbers, dates, and decisions that someone might search for later."""

        elif file_type in ['.xlsx', '.csv']:
            return f"""{base_instructions}

This is a DATA file. Please provide:

**METADATA SECTION:**
- Data Period: [What time period does this data cover?]
- Last Updated: [Any update dates mentioned]
- Data Source: [Where did this data come from?]

**CONTENT SECTION:**
- Description of what data this contains
- Table/sheet names and their purposes
- Key columns and data types
- Important metrics, totals, or trends with dates
- Geographic regions or business units
- Any time series or historical data patterns

**SEARCHABLE SUMMARY:**
Create a summary focusing on what questions this data could answer and when it's from.

Include all dates and time periods prominently."""

        else:  # Documents (.pdf, .docx, .pptx, etc.)
            # Determine document context based on filename/folder
            doc_context = ""
            if any(term in file_name.lower() for term in ["meeting", "minutes", "mom", "agenda"]):
                doc_context = "This appears to be MEETING MINUTES. "
            elif any(term in file_name.lower() for term in ["presentation", "ppt", "slides"]):
                doc_context = "This appears to be a PRESENTATION. "
            elif any(term in file_name.lower() for term in ["report", "analysis", "summary"]):
                doc_context = "This appears to be a REPORT/ANALYSIS. "
            elif "meeting" in folder_path.lower():
                doc_context = "This is from a meeting folder - likely MEETING CONTENT. "
            
            return f"""{base_instructions}

This is a DOCUMENT file ({file_type}). {doc_context}Extract EVERY searchable detail:

**METADATA SECTION:**
- Document Date: [When was this document created/meeting held?]
- Meeting Date: [If this is meeting minutes, when was the meeting?]
- Key Participants: [Full names, titles, departments of ALL people involved]
- Document Type: [Meeting minutes, report, presentation, etc.]

**GRANULAR CONTENT EXTRACTION:**
- Meeting attendees with full names and titles
- Specific topics discussed with context
- Exact decisions made (WHO decided WHAT involving WHICH PRODUCTS/AMOUNTS)
- Action items with responsible parties and specific deadlines
- Product names, model numbers, SKUs, project codes mentioned
- Dollar amounts, quantities, percentages, metrics discussed
- Problems identified with specific impacts and root causes
- Solutions proposed with timelines and responsible parties
- Company names, vendors, suppliers, customers mentioned
- Locations, facilities, regions, markets discussed
- Dates mentioned (deadlines, launch dates, meeting dates, milestones)
- Budget allocations, cost centers, approval levels
- Performance metrics, KPIs, targets, variances

**MEETING MINUTES SPECIFIC:**
- Agenda items covered
- Motions made and voting results
- Disagreements or concerns raised
- Follow-up meetings scheduled
- Presentation topics and key findings
- Questions asked and answers provided

**SEARCHABLE SUMMARY:**
Create a comprehensive summary: "On [EXACT DATE], [FULL NAMES/TITLES] met to discuss [SPECIFIC TOPICS]. Key decisions: [PERSON] approved [SPECIFIC DECISION] involving [PRODUCTS/AMOUNTS]. Action items: [PERSON] to [SPECIFIC ACTION] by [DATE]. Problems discussed: [SPECIFIC ISSUES] with [IMPACT]. Budget items: [AMOUNTS] for [PURPOSES]. Next steps: [SPECIFIC ACTIONS] by [DATES]."

Include ALL names, numbers, dates, decisions, and action items that someone might search for when looking for specific information."""
    
    def _extract_dates_from_summary(self, summary_text: str) -> List[str]:
        """Extract dates from summary text using regex patterns."""
        import re
        
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',       # YYYY-MM-DD
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',  # DD Month YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, summary_text, re.IGNORECASE)
            dates.extend(matches)
        
        # Remove duplicates and return
        return list(set(dates))
    
    def _create_summary_filename(self, original_filename: str, folder_path: str, dates: List[str]) -> str:
        """Create a searchable filename for the summary."""
        # Clean filename
        base_name = Path(original_filename).stem
        
        # Add primary date if available
        date_prefix = ""
        if dates:
            # Try to find the most relevant date (first one found)
            primary_date = dates[0]
            try:
                # Try to parse and format consistently
                from dateutil import parser
                parsed_date = parser.parse(primary_date)
                date_prefix = f"{parsed_date.strftime('%Y-%m-%d')}_"
            except:
                # Fallback to original date string
                date_prefix = f"{primary_date.replace('/', '-')}_"
        
        # Add folder context
        folder_clean = folder_path.replace('/', '_').replace(' ', '_').lower()
        
        return f"{date_prefix}{folder_clean}_{base_name}_summary"
    
    def _create_searchable_content(self, summary_text: str, file_name: str, folder_path: str) -> str:
        """Create enhanced searchable content combining all metadata and extracting key terms."""
        import re
        
        # Extract key searchable terms from summary
        searchable_terms = []
        
        # Extract names (capitalized words, likely people/companies)
        names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', summary_text)
        searchable_terms.extend(names)
        
        # Extract dollar amounts
        dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?[KMB]?', summary_text)
        searchable_terms.extend(dollar_amounts)
        
        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', summary_text)
        searchable_terms.extend(percentages)
        
        # Extract dates
        dates = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', summary_text)
        dates.extend(re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', summary_text))
        searchable_terms.extend(dates)
        
        # Extract product codes/SKUs (alphanumeric patterns)
        product_codes = re.findall(r'\b[A-Z]{2,}-?\d+[A-Z]?\b', summary_text)
        searchable_terms.extend(product_codes)
        
        # Extract quantities with units
        quantities = re.findall(r'\b\d{1,3}(?:,\d{3})*\s+(?:units|pieces|items|tons|pounds|kg|lbs)\b', summary_text, re.IGNORECASE)
        searchable_terms.extend(quantities)
        
        # Clean filename for additional searchable terms
        clean_filename = file_name.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        searchable_parts = [
            f"Filename: {file_name}",
            f"Location: {folder_path}",
            f"Key Terms: {' | '.join(set(searchable_terms))}",
            f"Clean Filename: {clean_filename}",
            f"Summary: {summary_text}"
        ]
        
        return " | ".join(searchable_parts)
    
    def _ensure_dropbox_folder_exists(self, folder_path: str) -> bool:
        """Ensure Dropbox folder exists, create if needed."""
        try:
            from dbx_utils import _get_dbx_client
            import dropbox
            
            client = _get_dbx_client()
            
            # Try to get folder metadata
            try:
                client.files_get_metadata(folder_path)
                return True  # Folder exists
            except dropbox.exceptions.ApiError as e:
                if e.error.is_path_not_found():
                    # Folder doesn't exist, create it
                    try:
                        client.files_create_folder_v2(folder_path)
                        print(f"📁 Created Dropbox folder: {folder_path}")
                        return True
                    except dropbox.exceptions.ApiError as create_error:
                        print(f"❌ Failed to create folder {folder_path}: {create_error}")
                        return False
                else:
                    print(f"❌ Error checking folder {folder_path}: {e}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error ensuring folder exists: {e}")
            return False

    def _save_summary_to_dropbox(self, summary_data: Dict[str, Any]) -> bool:
        """Save individual summary to Dropbox."""
        if not DROPBOX_AVAILABLE:
            print("Dropbox not available - cannot save summary")
            return False
            
        try:
            from dbx_utils import _get_dbx_client
            import dropbox
            
            # Ensure the summaries folder exists
            if not self._ensure_dropbox_folder_exists(self.dropbox_summaries_folder):
                print(f"❌ Cannot create summaries folder: {self.dropbox_summaries_folder}")
                return False
            
            # Create filename for individual summary
            safe_filename = summary_data['summary_filename'] + '.json'
            dropbox_path = f"{self.dropbox_summaries_folder}/{safe_filename}"
            
            # Upload summary to Dropbox
            client = _get_dbx_client()
            
            # Convert datetime objects to strings for JSON serialization
            def datetime_handler(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            summary_json = json.dumps(summary_data, indent=2, ensure_ascii=False, default=datetime_handler)
            
            client.files_upload(
                summary_json.encode('utf-8'),
                dropbox_path,
                mode=dropbox.files.WriteMode.overwrite
            )
            
            print(f"✅ Saved summary: {safe_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save summary to Dropbox: {e}")
            return False
    
    def _save_master_index_to_dropbox(self, all_summaries: List[Dict[str, Any]]) -> bool:
        """Save master index of all summaries to Dropbox."""
        if not DROPBOX_AVAILABLE:
            print("Dropbox not available - cannot save master index")
            return False
            
        try:
            from dbx_utils import _get_dbx_client
            import dropbox
            
            # Ensure the summaries folder exists
            if not self._ensure_dropbox_folder_exists(self.dropbox_summaries_folder):
                print(f"❌ Cannot create summaries folder for master index")
                return False
            
            # Create master index
            master_index = {
                "last_updated": datetime.now().isoformat(),
                "total_documents": len(all_summaries),
                "summaries": all_summaries
            }
            
            # Upload to Dropbox
            client = _get_dbx_client()
            
            # Convert datetime objects to strings for JSON serialization
            def datetime_handler(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            index_json = json.dumps(master_index, indent=2, ensure_ascii=False, default=datetime_handler)
            
            client.files_upload(
                index_json.encode('utf-8'),
                self.master_index_path,
                mode=dropbox.files.WriteMode.overwrite
            )
            
            print(f"✅ Saved master index with {len(all_summaries)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save master index to Dropbox: {e}")
            return False
    
    def _load_master_index_from_dropbox(self) -> List[Dict[str, Any]]:
        """Load existing master index from Dropbox."""
        if not DROPBOX_AVAILABLE:
            return []
            
        try:
            from dbx_utils import read_file_bytes
            
            # Try to read existing master index
            index_bytes = read_file_bytes(self.master_index_path)
            if index_bytes:
                index_data = json.loads(index_bytes.decode('utf-8'))
                return index_data.get('summaries', [])
                
        except Exception as e:
            print(f"No existing master index found: {e}")
            
        return []
    
    def _summarize_document(self, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create summary for a single document."""
        file_name = file_info['name']
        file_path = file_info['path']
        file_ext = Path(file_name).suffix.lower()
        folder_path = file_info.get('folder', 'root')
        
        print(f"📄 Processing: {file_name} (in {folder_path})")
        
        try:
            # Upload file to OpenAI
            file_id = upload_dropbox_file_to_openai(file_path, purpose="assistants")
            if not file_id:
                print(f"❌ Failed to upload {file_name}")
                return None
            
            # Create thread and get summary
            thread = self.client.beta.threads.create()
            
            summary_prompt = self._create_summary_prompt(file_name, file_ext, folder_path)
            
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=summary_prompt
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion
            max_wait = 60  # 60 second timeout for summaries
            waited = 0
            while run.status not in ["completed", "failed", "cancelled", "expired"] and waited < max_wait:
                import time
                time.sleep(3)
                run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                waited += 3
            
            if run.status == "completed":
                # Get the summary
                messages = self.client.beta.threads.messages.list(thread_id=thread.id, order="desc")
                if messages.data:
                    latest_message = messages.data[0]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        content_block = latest_message.content[0]
                        if hasattr(content_block, 'text'):
                            summary_text = content_block.text.value
                            
                            # Extract dates from summary for better searchability
                            extracted_dates = self._extract_dates_from_summary(summary_text)
                            
                            # Create searchable filename for the summary
                            summary_filename = self._create_summary_filename(file_name, folder_path, extracted_dates)
                            
                            return {
                                "file_name": file_name,
                                "file_path": file_path,
                                "folder": folder_path,
                                "full_folder_path": file_info.get('folder', 'root'),  # Full subfolder path
                                "file_type": file_ext,
                                "summary": summary_text,
                                "summary_filename": summary_filename,
                                "extracted_dates": extracted_dates,
                                "processed_date": datetime.now().isoformat(),
                                "file_size": file_info.get('size', 0),
                                "file_modified": file_info.get('modified_time', ''),
                                "openai_file_id": file_id,
                                "searchable_content": self._create_searchable_content(summary_text, file_name, folder_path)
                            }
            
            print(f"⚠️ Processing timeout or failed for {file_name}")
            return None
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            return None
    
    def process_new_files(self) -> Dict[str, Any]:
        """Process all new or changed files in KB."""
        if not DROPBOX_AVAILABLE:
            return {"error": "Dropbox utilities not available"}
        
        if not self.assistant_id:
            return {"error": "OpenAI Assistant not configured"}
        
        print("🔍 Scanning for new files...")
        print(f"📁 KB Path: {self.kb_path}")
        print(f"📁 Data Path: {self.data_path}")
        print(f"📁 Summaries will be saved to: {self.dropbox_summaries_folder}")
        
        # Get all files
        candidates = list_kb_candidates(self.kb_path, self.data_path)
        all_files = candidates.get('kb_docs', []) + candidates.get('data_files', [])
        
        print(f"📄 Found {len(all_files)} total files to consider")
        
        new_files = []
        updated_files = []
        skipped_files = []
        faiss_skipped = []
        
        for file_info in all_files:
            file_path = file_info['path']
            
            # Skip files already in FAISS index
            if file_path in self.faiss_indexed_files:
                faiss_skipped.append(file_info)
                continue
            
            file_hash = self._calculate_file_hash(file_path)
            
            # Check if file is new or changed
            if file_path not in self.file_hashes:
                new_files.append(file_info)
            elif self.file_hashes[file_path] != file_hash:
                updated_files.append(file_info)
            else:
                skipped_files.append(file_info)
        
        print(f"📊 Found: {len(new_files)} new, {len(updated_files)} updated, {len(skipped_files)} unchanged, {len(faiss_skipped)} already in FAISS")
        
        # Process new and updated files
        processed_count = 0
        failed_count = 0
        all_summaries = []
        
        for i, file_info in enumerate(new_files + updated_files, 1):
            total_to_process = len(new_files) + len(updated_files)
            print(f"🔄 Processing {i}/{total_to_process}: {file_info['name']}")
            
            summary_data = self._summarize_document(file_info)
            
            if summary_data:
                # Add to index
                file_path = file_info['path']
                self.document_index[file_path] = summary_data
                self.file_hashes[file_path] = self._calculate_file_hash(file_path)
                
                # Save individual summary to Dropbox
                if self._save_summary_to_dropbox(summary_data):
                    all_summaries.append(summary_data)
                    processed_count += 1
                    print(f"✅ Processed and saved: {file_info['name']}")
                else:
                    print(f"⚠️ Processed but failed to save: {file_info['name']}")
                    failed_count += 1
            else:
                print(f"❌ Failed to process: {file_info['name']}")
                failed_count += 1
        
        # Load existing summaries from Dropbox and merge with new ones
        existing_summaries = self._load_master_index_from_dropbox()
        
        # Merge existing with new (avoid duplicates by file path)
        existing_paths = {s.get('file_path') for s in existing_summaries}
        for summary in all_summaries:
            if summary.get('file_path') not in existing_paths:
                existing_summaries.append(summary)
        
        # Save master index to Dropbox
        self._save_master_index_to_dropbox(existing_summaries)
        
        # Save updated local indexes (for hash tracking)
        self._save_document_index()
        self._save_file_hashes()
        
        return {
            "processed": processed_count,
            "failed": failed_count,
            "skipped": len(skipped_files),
            "faiss_skipped": len(faiss_skipped),
            "total_files": len(all_files),
            "index_size": len(self.document_index)
        }
    
    def search_summaries(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Advanced search through document summaries with intelligent query analysis."""
        import re
        from datetime import datetime, timedelta
        
        query_lower = query.lower()
        results = []
        
        # Analyze query type for better search strategy
        query_analysis = self._analyze_query(query_lower)
        
        # Load master index from Dropbox if available
        master_summaries = self._load_master_index_from_dropbox()
        all_summaries = master_summaries if master_summaries else list(self.document_index.values())
        
        for summary_data in all_summaries:
            summary_text = summary_data.get('summary', '').lower()
            searchable_content = summary_data.get('searchable_content', '').lower()
            file_name = summary_data.get('file_name', '').lower()
            folder = summary_data.get('folder', '').lower()
            
            # Advanced relevance scoring
            score = 0
            
            # Keyword matching with weights
            for word in query_lower.split():
                if len(word) > 2:  # Skip short words
                    if word in summary_text:
                        score += 3
                    if word in searchable_content:
                        score += 2
                    if word in file_name:
                        score += 2
                    if word in folder:
                        score += 1
            
            # Query-type specific scoring
            if query_analysis['type'] == 'people':
                # Boost documents with people names
                for person_term in query_analysis['people_terms']:
                    if person_term in searchable_content:
                        score += 5
                        
            elif query_analysis['type'] == 'financial':
                # Boost documents with financial data
                if any(term in searchable_content for term in ['$', 'budget', 'cost', 'revenue']):
                    score += 4
                    
            elif query_analysis['type'] == 'product':
                # Boost documents with product references
                for product_term in query_analysis['product_terms']:
                    if product_term in searchable_content:
                        score += 4
                        
            elif query_analysis['type'] == 'temporal':
                # Boost recent documents for time-based queries
                doc_date = summary_data.get('processed_date')
                if doc_date:
                    try:
                        doc_datetime = datetime.fromisoformat(doc_date.replace('Z', '+00:00'))
                        days_old = (datetime.now() - doc_datetime).days
                        if days_old < 30:
                            score += 3
                        elif days_old < 90:
                            score += 2
                    except:
                        pass
            
            # Folder priority boost
            folder_priority = summary_data.get('folder_priority', 6)
            if folder_priority <= 2:
                score += 2
            elif folder_priority == 3:
                score += 1
            
            if score > 0:
                result = summary_data.copy()
                result['relevance_score'] = score
                result['query_analysis'] = query_analysis
                results.append(result)
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def _analyze_query(self, query_lower: str) -> Dict[str, Any]:
        """Analyze query to determine search strategy."""
        analysis = {
            'type': 'general',
            'people_terms': [],
            'product_terms': [],
            'financial_terms': [],
            'temporal_terms': []
        }
        
        # Check for people-focused queries
        people_indicators = ['who', 'person', 'manager', 'director', 'vp', 'ceo', 'cfo', 'team', 'responsible', 'assigned']
        if any(term in query_lower for term in people_indicators):
            analysis['type'] = 'people'
            # Extract potential names (capitalized words)
            import re
            names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', query_lower.title())
            analysis['people_terms'] = names
        
        # Check for financial queries
        financial_indicators = ['budget', 'cost', 'revenue', 'profit', 'expense', 'financial', 'money', '$', 'price', 'investment']
        if any(term in query_lower for term in financial_indicators):
            analysis['type'] = 'financial'
            analysis['financial_terms'] = [term for term in financial_indicators if term in query_lower]
        
        # Check for product queries
        product_indicators = ['product', 'model', 'sku', 'item', 'widget', 'launch', 'inventory', 'manufacturing']
        if any(term in query_lower for term in product_indicators):
            analysis['type'] = 'product'
            analysis['product_terms'] = [term for term in product_indicators if term in query_lower]
        
        # Check for time-based queries
        temporal_indicators = ['recent', 'latest', 'last', 'quarterly', 'monthly', 'this year', 'q1', 'q2', 'q3', 'q4']
        if any(term in query_lower for term in temporal_indicators):
            analysis['type'] = 'temporal'
            analysis['temporal_terms'] = [term for term in temporal_indicators if term in query_lower]
        
        return analysis

# Test function
def test_kb_indexer():
    """Test the KB indexer."""
    kb_path = os.getenv('KB_DBX_PATH', '/Project_Root/06_LLM_Knowledge_Base')
    data_path = os.getenv('DATA_DBX_PATH', '/Project_Root/04_Data')
    
    indexer = KBIndexer(kb_path, data_path)
    result = indexer.process_new_files()
    
    print("\n📋 Indexing Results:")
    print(f"Processed: {result.get('processed', 0)}")
    print(f"Failed: {result.get('failed', 0)}")
    print(f"Skipped: {result.get('skipped', 0)}")
    print(f"Total Index Size: {result.get('index_size', 0)}")

if __name__ == "__main__":
    test_kb_indexer()
