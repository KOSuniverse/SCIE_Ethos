#!/usr/bin/env python3
"""
SCIE Ethos Sync Manager
Handles ongoing Dropbox → OpenAI Assistant synchronization
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import signal
import sys

# Import the sync functionality
try:
    from dropbox_sync import (
        init_dropbox,
        dbx_read_json,
        dbx_write_json,
        list_dropbox_files,
        file_hash,
        resolve_assistant_id,
        get_or_create_vector_store,
        attach_vector_store_to_assistant,
        _upload_batch_to_vector_store,
        should_exclude_file
    )
    SYNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Dropbox sync not available: {e}")
    SYNC_AVAILABLE = False

# Configuration
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL_SECONDS", "300"))  # 5 minutes default
MANIFEST_PATH = "/prompts/dropbox_manifest.json"
WATCH_FOLDERS = [
    "04_Data/00_Raw_Files",
    "04_Data/01_Cleansed_Files", 
    "06_LLM_Knowledge_Base",
    "04_Data/04_Metadata"
]

class SyncManager:
    def __init__(self):
        self.running = False
        self.last_sync = None
        self.manifest = {}
        self.assistant_id = None
        self.vector_store_id = None
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize(self):
        """Initialize the sync manager."""
        if not SYNC_AVAILABLE:
            raise RuntimeError("Dropbox sync functionality not available")
        
        print("🚀 Initializing SCIE Ethos Sync Manager...")
        
        # Initialize Dropbox connection
        self.dbx = init_dropbox()
        print("✅ Dropbox connection established")
        
        # Load or create manifest
        self.manifest = dbx_read_json(MANIFEST_PATH) or {}
        print(f"📋 Loaded manifest with {len(self.manifest)} tracked files")
        
        # Resolve assistant and vector store
        self.assistant_id = resolve_assistant_id()
        self.vector_store_id = get_or_create_vector_store()
        attach_vector_store_to_assistant(self.assistant_id, self.vector_store_id)
        print(f"🤖 Assistant {self.assistant_id} ready")
        print(f"🔗 Vector store {self.vector_store_id} attached")
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        print("✅ Sync manager initialized successfully")
    
    def scan_for_changes(self) -> Dict[str, List[str]]:
        """Scan for file changes in watched folders."""
        changes = {
            "new": [],
            "modified": [],
            "deleted": []
        }
        
        # Get current file list
        current_files = set()
        for folder in WATCH_FOLDERS:
            try:
                files = list_dropbox_files(self.dbx, folder)
                for file_meta in files:
                    current_files.add(file_meta.path_lower)
            except Exception as e:
                print(f"⚠️ Error scanning {folder}: {e}")
                continue
        
        # Check for new/modified files
        for file_path in current_files:
            if file_path not in self.manifest:
                changes["new"].append(file_path)
            else:
                # Check if file was modified
                try:
                    _, resp = self.dbx.files_download(file_path)
                    current_hash = file_hash(resp.content)
                    if current_hash != self.manifest[file_path]["hash"]:
                        changes["modified"].append(file_path)
                except Exception as e:
                    print(f"⚠️ Error checking {file_path}: {e}")
                    continue
        
        # Check for deleted files
        for file_path in self.manifest:
            if file_path not in current_files:
                changes["deleted"].append(file_path)
        
        return changes
    
    def sync_changes(self, changes: Dict[str, List[str]]):
        """Sync detected changes to OpenAI."""
        total_changes = sum(len(files) for files in changes.values())
        if total_changes == 0:
            return
        
        print(f"🔄 Syncing {total_changes} changes...")
        
        # Handle new and modified files
        files_to_upload = changes["new"] + changes["modified"]
        if files_to_upload:
            self._upload_files(files_to_upload)
        
        # Handle deleted files
        if changes["deleted"]:
            self._remove_deleted_files(changes["deleted"])
        
        # Update manifest
        self._update_manifest(changes)
        
        # Save manifest
        dbx_write_json(MANIFEST_PATH, self.manifest)
        
        print(f"✅ Sync complete: {len(changes['new'])} new, {len(changes['modified'])} modified, {len(changes['deleted'])} deleted")
    
    def _upload_files(self, file_paths: List[str]):
        """Upload files to OpenAI vector store."""
        batch_size = 5
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            file_tuples = []
            
            for file_path in batch:
                try:
                    # Check if file should be excluded
                    file_name = os.path.basename(file_path)
                    ext = os.path.splitext(file_name)[1].lower()
                    should_exclude, reason = should_exclude_file(file_path, file_name, ext)
                    
                    if should_exclude:
                        print(f"⏭️ Skipping {file_name}: {reason}")
                        continue
                    
                    # Download and hash file
                    _, resp = self.dbx.files_download(file_path)
                    content = resp.content
                    file_hash_val = file_hash(content)
                    
                    file_tuples.append((file_name, content))
                    
                    # Update manifest
                    self.manifest[file_path] = {
                        "hash": file_hash_val,
                        "last_sync": datetime.utcnow().isoformat() + "Z"
                    }
                    
                    print(f"📤 Queued {file_name} for upload")
                    
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
                    continue
            
            if file_tuples:
                try:
                    _upload_batch_to_vector_store(self.vector_store_id, file_tuples)
                    print(f"✅ Uploaded batch of {len(file_tuples)} files")
                except Exception as e:
                    print(f"❌ Batch upload failed: {e}")
    
    def _remove_deleted_files(self, deleted_paths: List[str]):
        """Remove deleted files from manifest."""
        for file_path in deleted_paths:
            if file_path in self.manifest:
                del self.manifest[file_path]
                print(f"🗑️ Removed {file_path} from manifest")
    
    def _update_manifest(self, changes: Dict[str, List[str]]):
        """Update manifest with sync timestamps."""
        now = datetime.utcnow().isoformat() + "Z"
        
        for file_path in changes["new"] + changes["modified"]:
            if file_path in self.manifest:
                self.manifest[file_path]["last_sync"] = now
    
    def run_sync_cycle(self):
        """Run one sync cycle."""
        try:
            changes = self.scan_for_changes()
            if any(changes.values()):
                self.sync_changes(changes)
                self.last_sync = datetime.utcnow()
            else:
                print("✅ No changes detected")
        except Exception as e:
            print(f"❌ Sync cycle failed: {e}")
    
    def run(self):
        """Main run loop."""
        self.running = True
        print(f"🔄 Starting sync manager (interval: {SYNC_INTERVAL}s)")
        
        while self.running:
            try:
                self.run_sync_cycle()
                
                # Wait for next cycle
                if self.running:
                    print(f"⏰ Next sync in {SYNC_INTERVAL} seconds...")
                    time.sleep(SYNC_INTERVAL)
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                if self.running:
                    print("⏰ Retrying in 60 seconds...")
                    time.sleep(60)
        
        print("👋 Sync manager stopped")

def main():
    """Main entry point."""
    try:
        manager = SyncManager()
        manager.initialize()
        manager.run()
    except Exception as e:
        print(f"❌ Failed to start sync manager: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
