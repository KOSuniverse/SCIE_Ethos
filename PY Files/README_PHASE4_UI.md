# Phase 4: Streamlit UI Enhancements

## Overview

Phase 4 implements comprehensive Streamlit UI enhancements for the SCIE Ethos LLM Assistant, including service level control, enhanced export functionality, improved confidence scoring, enhanced sources management, and a data needed panel.

## Key Features Implemented

### 1. Service Level Control
- **Service Level Selector**: Dropdown with options {90%, 95%, 97.5%, 99%}
- **Z-Score Mapping**: Automatic conversion to statistical z-scores
- **Session State Integration**: Persistent service level selection
- **Visual Indicators**: Service level badges with z-score display

### 2. Enhanced Confidence Scoring
- **RAVC Methodology**: Recency, Alignment, Variance, Coverage scoring
- **Service Level Integration**: Confidence adjusted by selected service level
- **Enhanced Badges**: High/Medium/Low confidence with color coding
- **Abstention Logic**: Automatic abstention for low-confidence responses

### 3. Multi-Format Export System
- **Supported Formats**: XLSX, Markdown, DOCX, PPTX
- **Dual Storage**: Dropbox (primary) + S3 (audit)
- **Structured Exports**: Multiple sheets, metadata, sources, confidence history
- **Batch Export**: Single-click export to all formats

### 4. Enhanced Sources Drawer
- **Clickable Citations**: Interactive source management
- **Source Categorization**: File sources, KB sources, data sources, external sources
- **Content Previews**: File content and data previews
- **Download Integration**: Direct file download capabilities

### 5. Data Needed Panel
- **Gap Tracking**: Identify and track data gaps
- **Requirement Management**: Track data requirements and dependencies
- **Priority System**: Low/Medium/High/Critical priority levels
- **Status Tracking**: Open/Resolved status management

## Architecture

### Component Structure
```
Phase 4 UI Components
├── confidence.py              # Enhanced confidence scoring
├── export_utils.py            # Multi-format export management
├── sources_drawer.py          # Enhanced sources display
├── data_needed_panel.py       # Data gaps and requirements
└── chat_ui.py                 # Enhanced main UI (modified)
```

### Data Flow
1. **User Input** → Service Level Control → Confidence Scoring
2. **Query Processing** → Sources Collection → Enhanced Display
3. **Export Generation** → Multi-Format Creation → Dual Storage
4. **Gap Tracking** → Data Requirements → Status Management

## Implementation Details

### Enhanced Confidence Scoring (`confidence.py`)

#### Core Functions
- `score_ravc()`: Basic RAVC scoring
- `score_confidence_enhanced()`: Service level integrated scoring
- `get_service_level_zscore()`: Service level to z-score conversion
- `should_abstain()`: Abstention logic

#### Service Level Mapping
```python
service_level_map = {
    0.90: 1.645,  # 90% confidence interval
    0.95: 1.960,  # 95% confidence interval
    0.975: 2.241, # 97.5% confidence interval
    0.99: 2.576   # 99% confidence interval
}
```

#### Confidence Formula
```python
# From orchestrator_rules.yaml
raw_score = 0.35 * recency + 0.25 * alignment + 0.25 * (1 - variance) + 0.15 * coverage
```

### Export Management (`export_utils.py`)

#### ExportManager Class
- **Initialization**: Service level and timestamp tracking
- **Format Support**: XLSX, MD, DOCX, PPTX with fallbacks
- **Storage Integration**: Dropbox primary + S3 audit
- **Batch Operations**: Export to all formats simultaneously

#### Export Data Structure
```python
export_data = {
    "messages": chat_messages,
    "sources": last_sources,
    "confidence_history": confidence_scores,
    "metadata": {
        "service_level": 0.95,
        "exported_at": timestamp,
        "conversation_id": "default"
    },
    "data_gaps": gaps_summary
}
```

### Sources Management (`sources_drawer.py`)

#### SourcesDrawer Class
- **Source Types**: File, KB, Data, External
- **Interactive Elements**: Expandable source cards
- **Content Previews**: File content and data previews
- **Download Integration**: Direct file access

#### Source Categories
1. **File Sources**: Assistant files with metadata
2. **KB Sources**: Knowledge base documents
3. **Data Sources**: Excel/CSV files with previews
4. **External Sources**: URLs and external references

### Data Needs Tracking (`data_needed_panel.py`)

#### DataNeededPanel Class
- **Gap Management**: Add, track, and resolve data gaps
- **Requirement Tracking**: Data requirements and dependencies
- **Priority System**: Four-level priority classification
- **Status Management**: Open/Resolved status tracking

#### Gap Structure
```python
gap = {
    "description": "Missing inventory aging data",
    "impact": "Cannot calculate obsolescence risk",
    "data_type": "Inventory",
    "priority": "High",
    "suggested_actions": ["Request from ERP", "Manual collection"],
    "status": "Open",
    "timestamp": "2024-01-15 10:30:00"
}
```

## Usage Examples

### Service Level Control
```python
# In Streamlit sidebar
service_level = st.selectbox(
    "Select Service Level",
    options=[0.90, 0.95, 0.975, 0.99],
    index=1,  # Default to 95%
    format_func=lambda x: f"{x:.1%}"
)

# Get corresponding z-score
z_score = get_service_level_zscore(service_level)
```

### Enhanced Confidence Scoring
```python
# Calculate confidence with service level
confidence_data = score_confidence_enhanced(
    recency=0.8,
    alignment=0.9,
    variance=0.2,
    coverage=0.8,
    service_level=0.95
)

# Access confidence components
score = confidence_data["score"]
badge = confidence_data["badge"]
z_score = confidence_data["z_score"]
should_abstain = confidence_data["abstain"]
```

### Export Management
```python
# Initialize export manager
export_manager = ExportManager(service_level=0.95)

# Export to specific format
xlsx_content = export_manager.export_to_xlsx(export_data)
export_manager.save_to_dropbox(xlsx_content, "export.xlsx", "xlsx")

# Export to all formats
results = export_manager.export_all_formats(export_data)
```

### Sources Display
```python
# Initialize sources drawer
sources_drawer = SourcesDrawer()

# Render sources inline
sources_drawer.render_inline_sources(sources, confidence_score)

# Render collapsible sources panel
sources_drawer.render_collapsible_sources(sources, confidence_score)
```

### Data Needs Tracking
```python
# Initialize data panel
data_panel = DataNeededPanel()

# Add data gap
data_panel.add_data_gap(
    description="Missing supplier lead time data",
    impact="Cannot calculate safety stock levels",
    data_type="Inventory",
    priority="High",
    actions="Request from procurement team"
)

# Get gaps summary
summary = data_panel.get_gaps_summary()
```

## Configuration

### Service Level Settings
- **90%**: Standard confidence (z=1.645)
- **95%**: High confidence (z=1.960) - **Default**
- **97.5%**: Very high confidence (z=2.241)
- **99%**: Maximum confidence (z=2.576)

### Export Settings
- **Primary Storage**: Dropbox (`Project_Root/05_Exports`)
- **Audit Storage**: S3 (`{prefix}/exports/{timestamp}/`)
- **File Naming**: `scie_ethos_export_{timestamp}.{format}`

### Confidence Thresholds
- **High**: ≥0.75 (Green badge)
- **Medium**: 0.55-0.74 (Yellow badge)
- **Low**: <0.55 (Red badge)
- **Abstention**: <0.52 (Automatic abstention)

## Testing

### Test Script
Run the comprehensive test suite:
```bash
python scripts/test_phase4_ui.py
```

### Test Coverage
- ✅ Enhanced confidence scoring
- ✅ Export utilities
- ✅ Sources drawer
- ✅ Data needed panel
- ✅ Component integration

### Test Results
The test script validates:
- Component initialization
- Functionality of all methods
- Data structure integrity
- Integration between components
- Error handling and fallbacks

## Dependencies

### Required Packages
- `streamlit>=1.36`: UI framework
- `pandas>=2.2`: Data manipulation
- `openpyxl>=3.1`: Excel file handling
- `python-docx>=1.1`: Word document creation
- `python-pptx>=0.6`: PowerPoint creation
- `boto3>=1.34`: S3 integration

### Optional Dependencies
- `python-docx`: DOCX export (graceful fallback if missing)
- `python-pptx`: PPTX export (graceful fallback if missing)
- `boto3`: S3 export (graceful fallback if missing)

## Integration Points

### Phase 1-3 Integration
- **Orchestrator**: Confidence scoring integration
- **Assistant Bridge**: Query processing and response handling
- **Data Pipeline**: Export data from ingestion results
- **Knowledge Base**: Source citation integration

### External Systems
- **Dropbox**: Primary file storage and export destination
- **S3**: Audit trail and backup storage
- **OpenAI API**: Model selection and query processing

## Performance Considerations

### Export Performance
- **Large Datasets**: Pagination and chunking for large exports
- **Memory Management**: Streaming export for memory efficiency
- **Async Processing**: Background export processing for large files

### UI Responsiveness
- **Lazy Loading**: Sources and data loaded on demand
- **Caching**: Confidence scores and metadata cached
- **Progressive Rendering**: UI elements rendered progressively

## Security Features

### Data Protection
- **PII Handling**: Automatic PII detection and masking
- **Access Control**: Service level based access restrictions
- **Audit Logging**: Complete export and access logging

### Export Security
- **File Validation**: Export content validation
- **Access Logging**: S3 audit trail for all exports
- **Error Handling**: Secure error messages without data leakage

## Future Enhancements

### Planned Features
- **Real-time Collaboration**: Multi-user data gap tracking
- **Advanced Analytics**: Confidence score trend analysis
- **Custom Export Templates**: User-defined export formats
- **Integration APIs**: REST API for external system integration

### Scalability Improvements
- **Database Backend**: Persistent storage for large datasets
- **Caching Layer**: Redis integration for performance
- **Microservices**: Component separation for horizontal scaling

## Troubleshooting

### Common Issues

#### Export Failures
- **Missing Dependencies**: Install required packages
- **Storage Permissions**: Check Dropbox and S3 access
- **File Size Limits**: Check platform file size restrictions

#### Confidence Scoring Issues
- **Invalid Service Level**: Ensure service level is in valid range
- **Missing Data**: Check that all RAVC components are provided
- **Calculation Errors**: Verify input data types and ranges

#### Sources Display Issues
- **Missing Sources**: Check source data structure
- **Rendering Errors**: Verify Streamlit version compatibility
- **File Access**: Check file permissions and paths

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support and Maintenance

### Documentation
- **API Reference**: Complete function documentation
- **Examples**: Usage examples and best practices
- **Troubleshooting**: Common issues and solutions

### Updates
- **Version Control**: Git-based version management
- **Change Log**: Detailed change tracking
- **Migration Guide**: Upgrade path documentation

---

**Phase 4 Status**: ✅ **COMPLETED**

All Phase 4 UI enhancements have been successfully implemented and tested. The Streamlit interface now provides comprehensive service level control, enhanced export capabilities, improved confidence scoring, enhanced sources management, and data needs tracking functionality.
