# Phase 5: Logging & Retention + QA Systems

## Overview

Phase 5 implements comprehensive logging, retention management, QA testing, and monitoring systems for the SCIE Ethos LLM Assistant. This phase provides enterprise-grade operational capabilities including turn-by-turn logging, automated data retention, acceptance testing, and real-time system monitoring.

## Key Features Implemented

### 1. Comprehensive Logging System
- **Turn-by-Turn Logging**: Complete logging of every user interaction
- **S3 Integration**: Automatic upload to S3 with structured metadata
- **Schema Validation**: Strict adherence to `s3_turn_log.schema.json`
- **Session Management**: Unique session IDs and turn counting
- **Local Backup**: JSONL files for local storage and analysis

### 2. Retention Policy Management
- **Automated Cleanup**: Scheduled deletion of expired data
- **Policy Enforcement**: Configurable retention periods for logs and exports
- **S3 Integration**: Automatic cleanup of expired S3 objects
- **Audit Trail**: Complete tracking of cleanup operations
- **Flexible Configuration**: YAML-based policy management

### 3. QA Testing Framework
- **Acceptance Test Suite**: Automated validation of system performance
- **Citation Validation**: Ensures proper source attribution
- **Confidence Scoring**: Validates response quality thresholds
- **Keyword Validation**: Checks for required content coverage
- **Comprehensive Reporting**: Detailed test results and recommendations

### 4. Monitoring Dashboard
- **Real-Time Metrics**: Live system performance indicators
- **Confidence Trends**: Visual confidence score analysis
- **System Health**: Component health scoring and alerts
- **Quick Actions**: One-click system maintenance operations
- **Export Capabilities**: Metrics export for external analysis

## Architecture

### Component Structure
```
Phase 5 Systems
├── logging_system.py           # Core logging and retention
├── qa_framework.py            # QA testing and validation
├── monitoring_dashboard.py     # Real-time monitoring UI
└── test_phase5_systems.py     # Comprehensive test suite
```

### Data Flow
1. **User Interaction** → TurnLogger → S3 + Local Storage
2. **Retention Manager** → Policy Enforcement → Automated Cleanup
3. **QA Framework** → Test Execution → Validation → Reporting
4. **Monitoring Dashboard** → Real-Time Analytics → Alerts

## Implementation Details

### Logging System (`logging_system.py`)

#### TurnLogger Class
- **Session Management**: Unique session IDs and turn counting
- **Schema Compliance**: Strict validation against `s3_turn_log.schema.json`
- **Dual Storage**: Local JSONL files + S3 upload
- **Metadata Enrichment**: Automatic z-score calculation and validation

#### RetentionManager Class
- **Policy Loading**: YAML-based configuration management
- **S3 Cleanup**: Automatic deletion of expired objects
- **Audit Logging**: Complete tracking of cleanup operations
- **Flexible Retention**: Configurable periods for different data types

#### AnalyticsEngine Class
- **Trend Analysis**: Confidence score trend calculation
- **Performance Metrics**: Comprehensive system performance analysis
- **Recommendations**: AI-powered system improvement suggestions
- **Alert Generation**: Automatic issue detection and notification

### QA Framework (`qa_framework.py`)

#### QATestRunner Class
- **Test Suite Loading**: YAML-based acceptance test configuration
- **Automated Execution**: Batch test execution with validation
- **Response Validation**: Multi-criteria response quality assessment
- **Citation Checking**: Source attribution and relevance validation
- **Comprehensive Reporting**: Detailed test results and analysis

#### Test Validation Components
- **Citation Validation**: Country, sheet type, and source filtering
- **Keyword Validation**: Required content coverage checking
- **Confidence Validation**: Score threshold enforcement
- **Response Quality**: Multi-dimensional quality assessment

### Monitoring Dashboard (`monitoring_dashboard.py`)

#### MonitoringDashboard Class
- **Real-Time Metrics**: Live performance indicators
- **Interactive Charts**: Plotly-based data visualization
- **System Health**: Component health scoring and alerts
- **Quick Actions**: One-click system maintenance
- **Export Capabilities**: Metrics export for external analysis

#### Dashboard Sections
- **Performance Metrics**: Session turns, duration, confidence
- **Confidence Trends**: Visual trend analysis and forecasting
- **System Health**: Component health scoring and issue detection
- **Quick Actions**: Cleanup, reporting, and testing operations
- **Alerts & Notifications**: Real-time system alerts and warnings

## Usage Examples

### Logging System Usage
```python
from logging_system import TurnLogger, RetentionManager

# Initialize logger
logger = TurnLogger()

# Log a turn
turn_log = logger.log_turn(
    question="What is our inventory aging?",
    intent="inventory_analysis",
    sources=["inventory_data.xlsx"],
    confidence=0.85,
    model_used="gpt-4o",
    tokens=150,
    cost=0.003,
    service_level=0.95
)

# Get session summary
session_summary = logger.get_session_summary()

# Initialize retention manager
retention_manager = RetentionManager()

# Clean up expired data
cleanup_results = retention_manager.cleanup_expired_data()
```

### QA Testing Usage
```python
from qa_framework import QATestRunner

# Initialize QA runner
qa_runner = QATestRunner()

# Run all tests
results = qa_runner.run_all_tests(use_assistant=True)

# View test results
print(f"Tests passed: {results['test_summary']['passed']}/{results['test_summary']['total']}")
print(f"Overall score: {results['test_summary']['overall_score']:.1%}")

# Save test report
report_path = qa_runner.save_test_report(results)
```

### Monitoring Dashboard Usage
```python
from monitoring_dashboard import MonitoringDashboard

# Initialize dashboard
dashboard = MonitoringDashboard()

# Render dashboard (in Streamlit)
dashboard.render_dashboard()

# Get system health metrics
health_metrics = dashboard._get_system_health_metrics()

# Get system alerts
alerts = dashboard._get_system_alerts()

# Export dashboard metrics
dashboard._export_dashboard_metrics()
```

## Configuration

### Retention Policy Configuration
```yaml
# configs/retention_policy.yaml
s3:
  logs:    { prefix: "project-root/logs/",    retention_days: 365 }
  exports: { prefix: "project-root/exports/", retention_days: 180 }
dropbox:
  exports: { base_path: "/Apps/Ethos LLM/Project_Root/03_Summaries" }
```

### Acceptance Test Configuration
```yaml
# qa/acceptance_suite.yaml
tests:
  - id: inv_us_aging_insights
    ask: "Show US inventory by value and aging. Real insights."
    expect: 
      citations: { country: "US", sheet_type_any: ["inventory","eo"] }
      confidence_min: 0.70
```

### Environment Variables
```bash
# Required for S3 integration
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET=your_bucket_name

# Optional configuration
EMBED_MODEL=text-embedding-3-small
KB_CHUNK_TOKENS=700
KB_CHUNK_OVERLAP_TOKENS=100
```

## Testing

### Test Script
Run the comprehensive test suite:
```bash
python scripts/test_phase5_systems.py
```

### Test Coverage
- ✅ Logging system functionality
- ✅ Retention policy management
- ✅ QA testing framework
- ✅ Monitoring dashboard
- ✅ Component integration
- ✅ Data persistence
- ✅ Error handling

### Test Results
The test script validates:
- Component initialization and configuration
- Core functionality of all methods
- Data structure integrity and validation
- Integration between components
- Error handling and edge cases
- Data persistence and file operations

## Dependencies

### Required Packages
- `streamlit>=1.36`: UI framework
- `boto3>=1.34`: AWS S3 integration
- `pandas>=2.2`: Data analysis and manipulation
- `plotly>=5.18`: Interactive data visualization
- `pyyaml>=6.0`: YAML configuration parsing

### Optional Dependencies
- `boto3`: S3 integration (graceful fallback if missing)
- `pandas`: Advanced analytics (graceful fallback if missing)
- `plotly`: Interactive charts (graceful fallback if missing)

## Integration Points

### Phase 1-4 Integration
- **Orchestrator**: Logging integration for turn tracking
- **Assistant Bridge**: Query logging and response tracking
- **Confidence Scoring**: Integration with logging system
- **Export System**: Retention policy enforcement
- **Knowledge Base**: QA testing validation

### External Systems
- **AWS S3**: Log storage and retention management
- **Dropbox**: Export retention policy enforcement
- **Streamlit**: Real-time monitoring dashboard
- **OpenAI API**: Model usage tracking and cost analysis

## Performance Considerations

### Logging Performance
- **Batch Operations**: Efficient S3 upload batching
- **Local Caching**: Local storage for immediate access
- **Async Processing**: Background S3 operations
- **Compression**: Efficient log file storage

### Analytics Performance
- **Incremental Processing**: Delta-based trend analysis
- **Caching**: Performance metric caching
- **Lazy Loading**: On-demand data loading
- **Pagination**: Large dataset handling

## Security Features

### Data Protection
- **PII Handling**: Automatic PII detection and masking
- **Access Control**: S3 bucket policy enforcement
- **Audit Logging**: Complete operation tracking
- **Encryption**: S3 server-side encryption

### Logging Security
- **Schema Validation**: Strict input validation
- **Access Logging**: Complete access audit trail
- **Error Handling**: Secure error messages
- **Data Sanitization**: Input sanitization and validation

## Monitoring and Alerting

### System Health Metrics
- **Overall Health Score**: Composite system health indicator
- **Component Health**: Individual component performance scores
- **Trend Analysis**: Performance trend identification
- **Issue Detection**: Automatic problem identification

### Alert System
- **Confidence Alerts**: Low confidence response notifications
- **Performance Alerts**: System performance degradation warnings
- **Retention Alerts**: Data cleanup overdue notifications
- **Error Alerts**: System error and failure notifications

## Operational Procedures

### Daily Operations
1. **System Health Check**: Review dashboard health metrics
2. **Alert Review**: Check for active system alerts
3. **Performance Review**: Analyze confidence trends
4. **Log Review**: Check for unusual activity patterns

### Weekly Operations
1. **Retention Cleanup**: Run automated data cleanup
2. **Performance Analysis**: Generate weekly performance reports
3. **QA Test Execution**: Run acceptance test suite
4. **Trend Analysis**: Analyze weekly performance trends

### Monthly Operations
1. **Comprehensive Review**: Full system performance analysis
2. **Policy Review**: Retention policy effectiveness assessment
3. **Capacity Planning**: Storage and performance capacity analysis
4. **Improvement Planning**: System enhancement recommendations

## Troubleshooting

### Common Issues

#### Logging Issues
- **S3 Upload Failures**: Check AWS credentials and permissions
- **Schema Validation Errors**: Verify log data structure
- **Local Storage Issues**: Check file permissions and disk space

#### Retention Issues
- **Cleanup Failures**: Verify S3 bucket access and policies
- **Policy Loading Errors**: Check YAML syntax and file permissions
- **Expired Data Not Cleaned**: Verify retention policy configuration

#### QA Testing Issues
- **Test Execution Failures**: Check test configuration and dependencies
- **Validation Errors**: Verify test expectation format
- **Report Generation Issues**: Check file permissions and disk space

#### Dashboard Issues
- **Rendering Errors**: Verify Streamlit version compatibility
- **Data Loading Issues**: Check log file accessibility
- **Chart Generation Errors**: Verify plotly installation

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks
Run system health checks:
```python
from monitoring_dashboard import MonitoringDashboard

dashboard = MonitoringDashboard()
health = dashboard._get_system_health_metrics()
alerts = dashboard._get_system_alerts()

print(f"System Health: {health['overall_score']:.1%}")
print(f"Active Alerts: {len(alerts)}")
```

## Future Enhancements

### Planned Features
- **Real-Time Streaming**: Live log streaming and analysis
- **Advanced Analytics**: Machine learning-based trend prediction
- **Custom Dashboards**: User-configurable monitoring views
- **Integration APIs**: REST API for external system integration

### Scalability Improvements
- **Database Backend**: Persistent storage for large datasets
- **Distributed Logging**: Multi-node logging infrastructure
- **Caching Layer**: Redis integration for performance
- **Microservices**: Component separation for horizontal scaling

## Support and Maintenance

### Documentation
- **API Reference**: Complete function documentation
- **Configuration Guide**: Setup and configuration instructions
- **Troubleshooting**: Common issues and solutions
- **Operational Procedures**: Daily, weekly, and monthly tasks

### Updates
- **Version Control**: Git-based version management
- **Change Log**: Detailed change tracking
- **Migration Guide**: Upgrade path documentation
- **Release Notes**: Feature and bug fix documentation

---

**Phase 5 Status**: ✅ **COMPLETED**

All Phase 5 systems have been successfully implemented and tested. The SCIE Ethos system now provides comprehensive logging, retention management, QA testing, and real-time monitoring capabilities for enterprise-grade operations.
