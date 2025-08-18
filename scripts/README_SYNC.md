# SCIE Ethos Dropbox → OpenAI Assistant Sync

This directory contains the scripts for synchronizing Dropbox files with the OpenAI Assistant, enabling File Search + Code Interpreter capabilities.

## Overview

The sync system creates a bridge between Dropbox storage and OpenAI's Assistant API:

- **Documents (PDF, DOC, TXT, MD)**: Uploaded to OpenAI's vector store for File Search
- **Data files (Excel, CSV)**: Available via Code Interpreter from Dropbox
- **Automatic sync**: Continuous monitoring and syncing of file changes

## Architecture

```
Dropbox → Sync Manager → OpenAI Assistant
    ↓           ↓           ↓
  Raw Files  Manifest   Vector Store
  Excel/CSV  Tracking   File Search
  Documents  Changes    Code Interpreter
```

## Files

### Core Sync
- **`dropbox_sync.py`** - Main sync script with assistant creation
- **`sync_manager.py`** - Continuous sync manager for ongoing updates
- **`test_sync.py`** - Test suite for sync functionality

### Configuration
- **`prompts/assistant.json`** - Assistant metadata (auto-generated)
- **`prompts/instructions_master.yaml`** - Assistant instructions source

## Setup

### 1. Environment Variables

Set these required environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export DROPBOX_APP_KEY="your-dropbox-app-key"
export DROPBOX_APP_SECRET="your-dropbox-app-secret"
export DROPBOX_REFRESH_TOKEN="your-dropbox-refresh-token"

# Optional
export DROPBOX_ROOT="Project_Root"  # Your Dropbox project folder
export AUTO_CREATE_ASSISTANT="true"  # Auto-create assistant if missing
export SYNC_INTERVAL_SECONDS="300"   # Sync interval (5 minutes)
```

### 2. Dropbox App Setup

1. Create a Dropbox app at [Dropbox App Console](https://www.dropbox.com/developers/apps)
2. Set permissions to "Full Dropbox access"
3. Generate refresh token

### 3. Initial Sync

Run the initial sync to create the assistant and upload files:

```bash
# Create assistant and sync all files
python scripts/dropbox_sync.py

# Sync according to path contract structure
python scripts/dropbox_sync.py path-contract
```

### 4. Continuous Sync

Start the sync manager for ongoing file monitoring:

```bash
python scripts/sync_manager.py
```

## Usage

### One-time Sync

```bash
# Sync all files
python scripts/dropbox_sync.py

# Sync specific folders
python scripts/dropbox_sync.py path-contract
```

### Continuous Sync

```bash
# Start sync manager (default: 5-minute intervals)
python scripts/sync_manager.py

# Custom interval
export SYNC_INTERVAL_SECONDS="60"
python scripts/sync_manager.py
```

### Testing

```bash
# Run test suite
python scripts/test_sync.py
```

## File Handling

### Uploaded to OpenAI File Search
- **PDFs**: SOPs, policies, reports
- **Documents**: Word, PowerPoint, text files
- **Markdown**: Documentation, notes

### Available via Code Interpreter
- **Excel files**: Inventory, WIP, E&O data
- **CSV files**: Cleaned data, exports
- **Generated artifacts**: Charts, summaries

### Excluded from Sync
- Generated charts and summaries
- Log files
- Python cache files
- Temporary files

## Path Contract Integration

The sync respects the path contract structure:

```
Project_Root/
├── 04_Data/
│   ├── 00_Raw_Files/          # Raw data (Code Interpreter)
│   ├── 01_Cleansed_Files/     # Cleaned data (Code Interpreter)
│   ├── 02_EDA_Charts/         # Generated charts (excluded)
│   ├── 03_Summaries/          # Generated summaries (excluded)
│   ├── 04_Metadata/           # Metadata files (File Search)
│   └── 05_Merged_Comparisons/ # Generated comparisons (excluded)
├── 05_Exports/                 # Export files (Code Interpreter)
├── 06_Logs/                    # Log files (excluded)
└── 06_LLM_Knowledge_Base/     # Knowledge base (File Search)
```

## Monitoring

### Sync Status

The sync manager provides real-time status:

```
🔄 Syncing 3 changes...
📤 Queued inventory_q1.xlsx for upload
📤 Queued policy_sop.pdf for upload
📤 Queued wip_aging.csv for upload
✅ Uploaded batch of 3 files
✅ Sync complete: 2 new, 1 modified, 0 deleted
```

### Logs

Sync activity is logged to:
- Console output
- Dropbox manifest (`/prompts/dropbox_manifest.json`)
- Vector store metadata (`/prompts/vector_store.json`)

## Troubleshooting

### Common Issues

1. **Missing environment variables**
   - Check all required variables are set
   - Use `test_sync.py` to verify

2. **Dropbox connection failed**
   - Verify app key, secret, and refresh token
   - Check app permissions

3. **OpenAI API errors**
   - Verify API key is valid
   - Check API quota and limits

4. **File upload failures**
   - Check file size limits
   - Verify file formats are supported

### Debug Mode

Enable verbose logging:

```bash
export PYTHONPATH="."
python -u scripts/sync_manager.py
```

## Security

- API keys are stored in environment variables
- No sensitive data is logged
- File access is limited to specified folders
- Manifest files contain only file hashes and metadata

## Performance

- **Batch uploads**: Files uploaded in batches of 5
- **Incremental sync**: Only changed files are processed
- **Hash tracking**: File changes detected via SHA-256 hashes
- **Efficient polling**: Configurable sync intervals

## Integration

The sync system integrates with:

- **Streamlit UI**: Via `assistant_bridge.py`
- **Orchestrator**: File availability for analysis
- **Knowledge Base**: Document search capabilities
- **Data Analysis**: Excel/CSV access via Code Interpreter
