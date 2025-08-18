#!/usr/bin/env python3
"""
Debug script to test orchestrator flow
"""
import sys
from pathlib import Path

# Add PY Files to path
sys.path.append(str(Path(__file__).parent / "PY Files"))

try:
    from orchestrator import answer_question, classify_user_intent
    print("✅ Successfully imported orchestrator functions")
    
    # Test intent classification
    test_question = "Show me the top products by revenue"
    intent_result = classify_user_intent(test_question)
    print(f"✅ Intent classification working: {intent_result}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Runtime error: {e}")
    import traceback
    traceback.print_exc()
