#!/usr/bin/env python3
"""
Test script for Phase 5: Logging & Retention + QA Systems
Tests all new components: logging system, retention management, QA framework, and monitoring dashboard.
"""

import sys
import os
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add PY Files to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "PY Files"))

def test_logging_system():
    """Test comprehensive logging system."""
    print("🧪 Testing Logging System...")

    try:
        from logging_system import TurnLogger, RetentionManager, AnalyticsEngine

        # Test TurnLogger initialization
        logger = TurnLogger()
        assert logger.session_id is not None
        assert logger.turn_count == 0
        print("✅ TurnLogger initialization works")

        # Test turn logging
        turn_log = logger.log_turn(
            question="Test question",
            intent="test_intent",
            sources=["test_source.pdf"],
            confidence=0.85,
            model_used="gpt-4o",
            tokens=150,
            cost=0.003,
            service_level=0.95
        )

        assert turn_log["question"] == "Test question"
        assert turn_log["intent"] == "test_intent"
        assert turn_log["confidence"] == 0.85
        assert turn_log["z_score"] == 1.960  # 95% service level
        assert logger.turn_count == 1
        print("✅ Turn logging works")

        # Test session summary
        session_summary = logger.get_session_summary()
        assert session_summary["turn_count"] == 1
        assert session_summary["session_id"] == logger.session_id
        print("✅ Session summary works")

        # Test RetentionManager
        retention_manager = RetentionManager()
        assert retention_manager.retention_policy is not None
        print("✅ RetentionManager initialization works")

        # Test retention summary
        retention_summary = retention_manager.get_retention_summary()
        assert "policy" in retention_summary
        assert "last_cleanup" in retention_summary
        print("✅ Retention summary works")

        # Test AnalyticsEngine
        analytics_engine = AnalyticsEngine()
        assert analytics_engine.project_root is not None
        print("✅ AnalyticsEngine initialization works")

        # Test confidence trends (should work even without data)
        trends = analytics_engine.analyze_confidence_trends(days=7)
        assert isinstance(trends, dict)
        print("✅ Analytics engine works")

        return True

    except Exception as e:
        print(f"❌ Logging system test failed: {e}")
        return False

def test_qa_framework():
    """Test QA testing framework."""
    print("🧪 Testing QA Framework...")

    try:
        from qa_framework import QATestRunner

        # Test QATestRunner initialization
        qa_runner = QATestRunner()
        assert qa_runner.acceptance_suite is not None
        assert len(qa_runner.acceptance_suite["tests"]) > 0
        print("✅ QATestRunner initialization works")

        # Test acceptance suite loading
        tests = qa_runner.acceptance_suite["tests"]
        assert any(test["id"] == "inv_us_aging_insights" for test in tests)
        assert any(test["id"] == "par_now_q2_2026_single" for test in tests)
        assert any(test["id"] == "policy_workflow_kb" for test in tests)
        print("✅ Acceptance suite loading works")

        # Test test result structure
        test_result = qa_runner._run_single_test(tests[0], use_assistant=False)
        assert "test_id" in test_result
        assert "question" in test_result
        assert "response" in test_result
        assert "passed" in test_result
        print("✅ Test execution structure works")

        # Test validation methods
        validation_result = qa_runner._validate_response(
            {
                "answer": "Test answer with inventory and aging",
                "citations": [{"type": "file", "metadata": {"country": "US", "sheet_type": "inventory"}}],
                "includes_keywords": ["inventory", "aging"],
                "confidence": 0.8
            },
            {
                "citations": {"country": "US", "sheet_type_any": ["inventory", "eo"]},
                "includes": ["inventory", "aging"],
                "confidence_min": 0.7
            }
        )

        assert validation_result["passed"] == True
        print("✅ Response validation works")

        # Test citation validation
        citation_result = qa_runner._validate_citations(
            [{"type": "file", "metadata": {"country": "US", "sheet_type": "inventory"}}],
            {"country": "US", "sheet_type_any": ["inventory"]}
        )
        assert citation_result["passed"] == True
        print("✅ Citation validation works")

        # Test keyword validation
        keyword_result = qa_runner._validate_keywords(
            ["inventory", "aging", "value"],
            ["inventory", "aging"]
        )
        assert keyword_result["passed"] == True
        assert keyword_result["match_rate"] == 1.0
        print("✅ Keyword validation works")

        # Test confidence validation
        confidence_result = qa_runner._validate_confidence(0.8, 0.7)
        assert confidence_result["passed"] == True
        assert confidence_result["margin"] == 0.1
        print("✅ Confidence validation works")

        return True

    except Exception as e:
        print(f"❌ QA framework test failed: {e}")
        return False

def test_monitoring_dashboard():
    """Test monitoring dashboard components."""
    print("🧪 Testing Monitoring Dashboard...")

    try:
        from monitoring_dashboard import MonitoringDashboard

        # Test dashboard initialization
        dashboard = MonitoringDashboard()
        assert dashboard.logger is not None
        assert dashboard.retention_manager is not None
        assert dashboard.analytics_engine is not None
        assert dashboard.qa_runner is not None
        print("✅ Dashboard initialization works")

        # Test system health metrics
        health_metrics = dashboard._get_system_health_metrics()
        assert "overall_score" in health_metrics
        assert "components" in health_metrics
        assert "recent_issues" in health_metrics
        print("✅ System health metrics work")

        # Test system alerts
        alerts = dashboard._get_system_alerts()
        assert isinstance(alerts, list)
        print("✅ System alerts work")

        # Test duration formatting
        duration = dashboard._format_duration("2024-01-15T10:00:00Z")
        assert isinstance(duration, str)
        print("✅ Duration formatting works")

        # Test that dashboard has all required methods
        required_methods = [
            "render_dashboard",
            "_render_performance_metrics",
            "_render_confidence_trends",
            "_render_system_health",
            "_render_quick_actions",
            "_render_alerts",
            "_render_retention_status"
        ]

        for method in required_methods:
            assert hasattr(dashboard, method)
        print("✅ All required dashboard methods are present")

        return True

    except Exception as e:
        print(f"❌ Monitoring dashboard test failed: {e}")
        return False

def test_integration():
    """Test integration between Phase 5 components."""
    print("🧪 Testing Component Integration...")

    try:
        # Test that all components can work together
        from logging_system import TurnLogger, RetentionManager, AnalyticsEngine
        from qa_framework import QATestRunner
        from monitoring_dashboard import MonitoringDashboard

        # Create a temporary project root for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            logger = TurnLogger(temp_dir)
            retention_manager = RetentionManager(temp_dir)
            analytics_engine = AnalyticsEngine(temp_dir)
            qa_runner = QATestRunner(temp_dir)
            dashboard = MonitoringDashboard(temp_dir)

            # Test logging integration
            turn_log = logger.log_turn(
                question="Integration test question",
                intent="integration_test",
                sources=["test_integration.pdf"],
                confidence=0.9,
                service_level=0.95
            )

            # Test retention integration
            retention_summary = retention_manager.get_retention_summary()
            assert retention_summary is not None

            # Test analytics integration
            trends = analytics_engine.analyze_confidence_trends(days=1)
            assert isinstance(trends, dict)

            # Test QA integration
            test_suite = qa_runner.acceptance_suite
            assert test_suite is not None

            # Test dashboard integration
            health_metrics = dashboard._get_system_health_metrics()
            assert health_metrics is not None

        print("✅ Component integration works")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_data_persistence():
    """Test data persistence and file operations."""
    print("🧪 Testing Data Persistence...")

    try:
        from logging_system import TurnLogger

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize logger with temp directory
            logger = TurnLogger(temp_dir)

            # Log several turns
            for i in range(3):
                logger.log_turn(
                    question=f"Test question {i+1}",
                    intent=f"test_intent_{i+1}",
                    sources=[f"test_source_{i+1}.pdf"],
                    confidence=0.8 + (i * 0.05),
                    service_level=0.95
                )

            # Check that logs directory was created
            logs_dir = Path(temp_dir) / "06_Logs"
            assert logs_dir.exists()

            # Check that log files were created
            log_files = list(logs_dir.glob("turn_logs_*.jsonl"))
            assert len(log_files) > 0

            # Check log file content
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) > 0
                    
                    # Parse JSON lines
                    for line in lines:
                        if line.strip():
                            log_entry = json.loads(line)
                            assert "question" in log_entry
                            assert "intent" in log_entry
                            assert "confidence" in log_entry

            print("✅ Data persistence works")

        return True

    except Exception as e:
        print(f"❌ Data persistence test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("🧪 Testing Error Handling...")

    try:
        from logging_system import TurnLogger, RetentionManager, AnalyticsEngine
        from qa_framework import QATestRunner

        # Test TurnLogger with invalid data
        logger = TurnLogger()
        
        # Test with missing required fields (should raise ValueError)
        try:
            logger.log_turn(
                question="",  # Empty question
                intent="",    # Empty intent
                sources=[],   # Empty sources
                confidence="invalid"  # Invalid confidence type
            )
            print("❌ Should have raised ValueError for invalid data")
            return False
        except ValueError:
            print("✅ Properly handles invalid data")

        # Test retention manager with missing S3
        retention_manager = RetentionManager()
        cleanup_results = retention_manager.cleanup_expired_data()
        assert isinstance(cleanup_results, dict)
        assert "logs_cleaned" in cleanup_results
        print("✅ Gracefully handles missing S3")

        # Test analytics engine with no data
        analytics_engine = AnalyticsEngine()
        trends = analytics_engine.analyze_confidence_trends(days=1)
        assert isinstance(trends, dict)
        print("✅ Gracefully handles no data")

        # Test QA runner with invalid test data
        qa_runner = QATestRunner()
        
        # Test validation with empty response
        validation_result = qa_runner._validate_response(
            {
                "answer": "",
                "citations": [],
                "includes_keywords": [],
                "confidence": 0.0
            },
            {
                "citations": {"country": "US"},
                "includes": ["inventory"],
                "confidence_min": 0.7
            }
        )
        
        assert validation_result["passed"] == False
        assert len(validation_result["errors"]) > 0
        print("✅ Properly handles empty/invalid responses")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all Phase 5 tests."""
    print("🚀 Starting Phase 5 Systems Tests...")
    print("=" * 60)

    tests = [
        ("Logging System", test_logging_system),
        ("QA Framework", test_qa_framework),
        ("Monitoring Dashboard", test_monitoring_dashboard),
        ("Component Integration", test_integration),
        ("Data Persistence", test_data_persistence),
        ("Error Handling", test_error_handling)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 40)

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Phase 5 systems are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
