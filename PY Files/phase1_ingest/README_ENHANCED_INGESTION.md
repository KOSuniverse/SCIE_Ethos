# Enhanced Data Ingestion Pipeline

The Enhanced Data Ingestion Pipeline is the core component of Phase 3 that produces all required artifacts according to `ingest_rules.yaml` and integrates with existing components.

## Overview

The enhanced pipeline extends the existing ingestion system to produce:
- **master_catalog.jsonl** - Comprehensive metadata catalog
- **eda_profile.json** - EDA analysis profiles
- **summary_card.md** - Human-readable data summaries
- **Knowledge base ingestion** - PDF/document processing

## Architecture

```
Raw Excel Files → Enhanced Pipeline → Artifacts
                    ↓
            [Existing Components]
            - pipeline.py (core processing)
            - sheet_utils.py (classification)
            - smart_cleaning.py (data cleaning)
            - enhanced_eda_system.py (analysis)
            - knowledgebase_builder.py (KB ingestion)
```

## Key Features

### 1. Ontology-Based Metadata Extraction
Extracts metadata according to the 4-tier ontology defined in `ingest_rules.yaml`:

- **Tier 0**: Core identifiers (ERP, country, income stream, sheet type)
- **Tier 1**: Business context (product family, business unit, market, channel, plant, etc.)
- **Tier 2**: Temporal context (fiscal period, snapshot date, granularity, horizon)
- **Tier 3**: Technical metadata (source file, version, language, PII detection)
- **Tier 4**: Analysis metadata (primary keys, joinability, lead times, data presence flags)

### 2. Intelligent Data Detection
Automatically detects:
- **ERP systems** from filename/content patterns
- **Countries** from location references
- **Data types** (inventory, WIP, raw materials, finished goods)
- **Currencies** and **units of measure**
- **Lead time statistics** when available
- **Aging bucket presence** for inventory analysis

### 3. Enhanced EDA Integration
Integrates with the existing EDA system to:
- Generate comprehensive data profiles
- Create business insights
- Produce visualization charts
- Store analysis results

### 4. Knowledge Base Ingestion
Processes documents according to `ingest_rules.yaml`:
- **PDF processing** with OCR support
- **Document classification** by folder patterns
- **Tagging** based on content and location
- **Vector embedding** for semantic search

### 5. Artifact Generation
Produces standardized outputs:
- **master_catalog.jsonl**: JSONL format for easy processing
- **eda_profile.json**: Structured EDA results
- **summary_card.md**: Human-readable summaries using templates

## Usage

### Basic Usage

```python
from phase1_ingest.enhanced_ingestion_pipeline import EnhancedIngestionPipeline

# Initialize pipeline
pipeline = EnhancedIngestionPipeline()

# Process single file
result = pipeline.process_file("data.xlsx")

# Process multiple files
results = pipeline.run_full_ingestion(["file1.xlsx", "file2.xlsx"])

# Emit artifacts
artifacts = pipeline.emit_artifacts()
```

### CLI Usage

```bash
# Setup environment and create sample data
python scripts/run_ingestion.py --setup

# Process specific files
python scripts/run_ingestion.py data1.xlsx data2.xlsx

# Process all files in raw data directory
python scripts/run_ingestion.py --raw-data-dir

# Include knowledge base ingestion
python scripts/run_ingestion.py --kb data.xlsx

# Verbose output
python scripts/run_ingestion.py --verbose data.xlsx
```

### Testing

```bash
# Run test suite
python scripts/test_ingestion.py

# Test specific functionality
python -c "
from phase1_ingest.enhanced_ingestion_pipeline import EnhancedIngestionPipeline
pipeline = EnhancedIngestionPipeline()
print('Pipeline initialized successfully')
"
```

## Configuration

### ingest_rules.yaml

The pipeline reads configuration from `prompts/ingest_rules.yaml`:

```yaml
ontology:
  tier0: ["erp", "country", "income_stream", "sheet_type"]
  tier1: ["product_family", "business_unit", "market", "channel"]
  # ... more tiers

artifacts:
  emit_master_catalog: true
  emit_eda_profile: true
  emit_summary_card: true

knowledge_base_ingest:
  include: true
  file_types: ["pdf"]
  ocr: true
```

### Path Contract

Output locations are controlled by `configs/path_contract.yaml`:

```yaml
base_data_dir: "Project_Root/04_Data"
folders:
  metadata: "04_Metadata"
  raw: "00_Raw_Files"
  cleansed: "01_Cleansed_Files"
```

## Output Artifacts

### 1. master_catalog.jsonl

Each line contains a JSON object with comprehensive metadata:

```json
{
  "filename": "inventory_q1.xlsx",
  "sheet_name": "Inventory",
  "erp": "SAP",
  "country": "US",
  "income_stream": "Inventory",
  "sheet_type": "inventory",
  "row_count": 1500,
  "col_count": 25,
  "currency": "USD",
  "uom": "PCS",
  "has_aging": true,
  "has_wip": false,
  "lt_mean": 45.2,
  "lt_std": 12.8,
  "join_keys_found": ["Part_Number", "Plant"],
  "join_score": 0.85,
  "processed_at": "2024-01-15T10:30:00Z"
}
```

### 2. eda_profile.json

Structured EDA results:

```json
{
  "version": "1.0",
  "generated_at": "2024-01-15T10:30:00Z",
  "profiles": {
    "inventory_q1.xlsx_Inventory": {
      "metadata": {"rows": 1500, "cols": 25},
      "chart_paths": ["charts/inventory_distribution.png"],
      "business_insights": {
        "aging_analysis": "30% of inventory is over 90 days old",
        "value_concentration": "Top 20% of parts represent 80% of value"
      }
    }
  }
}
```

### 3. summary_card.md

Human-readable summaries using templates:

```markdown
# Inventory - US

**Scope:** inventory — US — ERP SAP
**Time window:** Q1 2024 → Q1 2024 (quarterly)
**Rows/Cols:** 1500 / 25
**Language:** en | **Source:** inventory_q1.xlsx#Inventory

## Key facts
- On-hand: 45,000 (PCS), Value USD 2,250,000
- Aging buckets present: Yes
- Lead-time stats: μ=45.2 days, σ=12.8 days

## What this answers well
- Current inventory levels by part and location
- Aging analysis for obsolescence risk
- Value concentration analysis
- Lead time variability assessment

## Caveats / Missing
- Data quality assessment pending
- Business rules validation needed
- Cross-reference verification required
```

## Integration Points

### Existing Components

The enhanced pipeline integrates with:

1. **pipeline.py**: Core Excel processing and sheet handling
2. **sheet_utils.py**: Sheet classification and type detection
3. **smart_cleaning.py**: Data quality and cleaning operations
4. **enhanced_eda_system.py**: Comprehensive EDA analysis
5. **knowledgebase_builder.py**: Document ingestion and processing
6. **metadata_utils.py**: Metadata storage and retrieval

### Fallback Handling

When components are unavailable, the pipeline gracefully falls back to:
- Basic Excel reading without advanced processing
- Default metadata values
- Simple summary generation
- Error logging and reporting

## Error Handling

The pipeline implements comprehensive error handling:

- **File-level errors**: Logged but don't stop processing of other files
- **Sheet-level errors**: Individual sheet failures don't affect other sheets
- **Component failures**: Graceful degradation when dependencies are missing
- **Validation errors**: Clear reporting of data quality issues

## Performance Considerations

- **Batch processing**: Files processed sequentially to manage memory
- **Incremental updates**: Only changed files are reprocessed
- **Resource management**: EDA and KB processing can be disabled for faster processing
- **Parallel processing**: Future enhancement for large file sets

## Extensibility

The pipeline is designed for easy extension:

1. **New metadata fields**: Add to ontology tiers in `ingest_rules.yaml`
2. **Custom detectors**: Implement new `_detect_*` methods
3. **Additional artifacts**: Extend `emit_artifacts()` method
4. **Integration hooks**: Add new component integrations

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure PY Files directory is in Python path
2. **Missing dependencies**: Install required packages (pandas, numpy, yaml)
3. **Path issues**: Verify path contract configuration
4. **Permission errors**: Check write permissions for output directories

### Debug Mode

Enable verbose logging:

```bash
python scripts/run_ingestion.py --verbose data.xlsx
```

### Testing

Run the test suite to verify functionality:

```bash
python scripts/test_ingestion.py
```

## Future Enhancements

Planned improvements:

1. **Parallel processing** for large file sets
2. **Incremental updates** based on file modification times
3. **Real-time monitoring** of ingestion progress
4. **Advanced validation** rules and data quality scoring
5. **Integration** with external data quality tools
6. **Performance optimization** for very large datasets

## Contributing

When extending the pipeline:

1. **Follow existing patterns** for metadata extraction
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for new features
4. **Maintain backward compatibility** where possible
5. **Use graceful fallbacks** for missing dependencies
