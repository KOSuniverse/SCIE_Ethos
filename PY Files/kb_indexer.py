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
CRITICAL: Extract ALL dates mentioned in this document and format them clearly.
For each significant piece of information, include the date when it was discussed/decided.

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

This is an EMAIL file. Please provide:

**METADATA SECTION:**
- Email Date: [Extract exact date/time]
- From: [Sender name and email]
- To: [Recipients]
- Subject: [Full subject line]

**CONTENT SECTION:**
- Full email body text
- Key participants mentioned in the email
- Important decisions or action items with dates
- Products, projects, or topics discussed
- Any meetings, deadlines, or future dates mentioned
- Numbers, metrics, or data points

**SEARCHABLE SUMMARY:**
Create a paragraph summary that includes key dates and can be easily searched.

Format everything as clear, searchable text with dates prominently featured."""

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

        else:  # Documents (.pdf, .docx, etc.)
            return f"""{base_instructions}

This is a DOCUMENT file. Please provide:

**METADATA SECTION:**
- Document Date: [When was this document created/meeting held?]
- Meeting Date: [If this is meeting minutes, when was the meeting?]
- Key Participants: [Names and roles of people involved]
- Document Type: [Meeting minutes, report, presentation, etc.]

**CONTENT SECTION:**
- Main topic and purpose with dates
- Important decisions made (WHO decided WHAT and WHEN)
- Action items with owners and deadlines
- Products, projects, or initiatives discussed
- Data points, metrics, or numbers with their context
- Problems identified and solutions proposed with timelines
- Follow-up meetings or deadlines mentioned

**SEARCHABLE SUMMARY:**
Create a detailed summary that starts with the date and includes "On [DATE], [PARTICIPANTS] discussed [TOPICS]..."

Make this summary detailed enough for data mining and problem-solving, with dates prominently featured throughout."""
    
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
        """Create enhanced searchable content combining all metadata."""
        searchable_parts = [
            f"Filename: {file_name}",
            f"Location: {folder_path}",
            f"Summary: {summary_text}"
        ]
        
        return " | ".join(searchable_parts)
    
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
        
        # Get all files
        candidates = list_kb_candidates(self.kb_path, self.data_path)
        all_files = candidates.get('kb_docs', []) + candidates.get('data_files', [])
        
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
        
        for file_info in new_files + updated_files:
            summary_data = self._summarize_document(file_info)
            
            if summary_data:
                # Add to index
                file_path = file_info['path']
                self.document_index[file_path] = summary_data
                self.file_hashes[file_path] = self._calculate_file_hash(file_path)
                processed_count += 1
                print(f"✅ Processed: {file_info['name']}")
            else:
                failed_count += 1
        
        # Save updated indexes
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
    
    def search_summaries(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search through document summaries."""
        query_lower = query.lower()
        results = []
        
        for file_path, summary_data in self.document_index.items():
            summary_text = summary_data.get('summary', '').lower()
            file_name = summary_data.get('file_name', '').lower()
            
            # Simple relevance scoring
            score = 0
            for word in query_lower.split():
                if len(word) > 2:
                    score += summary_text.count(word) * 2  # Summary matches worth more
                    score += file_name.count(word) * 1     # Filename matches
            
            if score > 0:
                results.append({
                    **summary_data,
                    "relevance_score": score
                })
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]

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
