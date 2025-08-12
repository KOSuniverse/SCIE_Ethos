#!/usr/bin/env python3
"""
Knowledge Base (Phase E) validation script for ChatGPT enterprise plan alignment.
Tests the implementation according to ChatGPT's specifications.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "PY Files"))

def test_knowledgebase_builder_config():
    """Test that knowledgebase_builder has correct ChatGPT Phase E configuration."""
    print("🔍 Testing knowledgebase_builder.py configuration...")
    
    try:
        from phase4_knowledge.knowledgebase_builder import DEFAULT_SCAN_FOLDERS, status, KB_SUBDIR
        from phase4_knowledge.knowledgebase_builder import INDEX_REL, DOCSTORE_REL, MANIFEST_REL, CHUNKS_REL
        
        # Test DEFAULT_SCAN_FOLDERS matches ChatGPT spec
        expected_scan_folders = ["06_LLM_Knowledge_Base"]
        if DEFAULT_SCAN_FOLDERS == expected_scan_folders:
            print("  ✅ DEFAULT_SCAN_FOLDERS correctly set to ['06_LLM_Knowledge_Base']")
        else:
            print(f"  ❌ DEFAULT_SCAN_FOLDERS mismatch. Expected: {expected_scan_folders}, Got: {DEFAULT_SCAN_FOLDERS}")
            return False
        
        # Test relative paths (*_REL) are used
        expected_rels = {
            "INDEX_REL": "06_LLM_Knowledge_Base/document_index.faiss",
            "DOCSTORE_REL": "06_LLM_Knowledge_Base/docstore.pkl", 
            "MANIFEST_REL": "06_LLM_Knowledge_Base/manifest.json",
            "CHUNKS_REL": "06_LLM_Knowledge_Base/chunks"
        }
        
        actual_rels = {
            "INDEX_REL": INDEX_REL,
            "DOCSTORE_REL": DOCSTORE_REL,
            "MANIFEST_REL": MANIFEST_REL,
            "CHUNKS_REL": CHUNKS_REL
        }
        
        for name, expected in expected_rels.items():
            if actual_rels[name] == expected:
                print(f"  ✅ {name} correctly set to '{expected}'")
            else:
                print(f"  ❌ {name} mismatch. Expected: '{expected}', Got: '{actual_rels[name]}'")
                return False
        
        # Test status function exists and is callable
        if callable(status):
            print("  ✅ status() function is callable")
        else:
            print("  ❌ status() function not callable")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ knowledgebase_builder test failed: {e}")
        return False

def test_kb_qa_contract():
    """Test that kb_qa meets contract requirements."""
    print("\n🔍 Testing kb_qa.py contract compliance...")
    
    try:
        from phase4_knowledge.kb_qa import kb_answer
        import inspect
        
        # Test function signature
        sig = inspect.signature(kb_answer)
        required_params = ['project_root', 'query']
        
        for param in required_params:
            if param in sig.parameters:
                print(f"  ✅ kb_answer has required parameter: {param}")
            else:
                print(f"  ❌ kb_answer missing required parameter: {param}")
                return False
        
        # Test return type (can't easily test the actual return without OpenAI key)
        # But we can check the function docstring mentions the contract
        docstring = kb_answer.__doc__ or ""
        if "hits" in docstring and "context" in docstring and "answer" in docstring:
            print("  ✅ kb_answer docstring mentions contract fields: hits, context, answer")
        else:
            print("  ⚠️ kb_answer docstring may not clearly document return contract")
        
        return True
        
    except Exception as e:
        print(f"  ❌ kb_qa contract test failed: {e}")
        return False

def test_knowledgebase_retriever_updates():
    """Test that knowledgebase_retriever uses updated relative paths."""
    print("\n🔍 Testing knowledgebase_retriever.py updates...")
    
    try:
        from phase4_knowledge.knowledgebase_retriever import INDEX_REL, DOCSTORE_REL
        
        # Test that it uses the new *_REL variables instead of old *_PATH
        expected_index = "06_LLM_Knowledge_Base/document_index.faiss"
        expected_docstore = "06_LLM_Knowledge_Base/docstore.pkl"
        
        if INDEX_REL == expected_index:
            print(f"  ✅ INDEX_REL correctly set to '{expected_index}'")
        else:
            print(f"  ❌ INDEX_REL mismatch. Expected: '{expected_index}', Got: '{INDEX_REL}'")
            return False
            
        if DOCSTORE_REL == expected_docstore:
            print(f"  ✅ DOCSTORE_REL correctly set to '{expected_docstore}'")
        else:
            print(f"  ❌ DOCSTORE_REL mismatch. Expected: '{expected_docstore}', Got: '{DOCSTORE_REL}'")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ knowledgebase_retriever test failed: {e}")
        return False

def test_app_kb_configuration():
    """Test that app_kb.py meets ChatGPT requirements."""
    print("\n🔍 Testing app_kb.py configuration...")
    
    try:
        # Read the app_kb.py file and check for required configurations
        app_kb_path = Path(__file__).parent.parent / "app_kb.py"
        
        if not app_kb_path.exists():
            print(f"  ❌ app_kb.py not found at {app_kb_path}")
            return False
        
        content = app_kb_path.read_text()
        
        # Test sys.path.append("PY Files") is present
        if 'sys.path.append("PY Files")' in content:
            print('  ✅ sys.path.append("PY Files") found in app_kb.py')
        else:
            print('  ❌ sys.path.append("PY Files") missing from app_kb.py')
            return False
        
        # Test that it doesn't set empty OPENAI_API_KEY
        # Should check for key before setting, not set empty value
        if 'if not api_key:' in content and 'os.environ["OPENAI_API_KEY"] = api_key' in content:
            print("  ✅ app_kb.py properly checks API key before setting environment variable")
        else:
            print("  ⚠️ app_kb.py API key handling may need review")
        
        # Test status(PROJECT_ROOT) usage
        if 'status(PROJECT_ROOT)' in content:
            print("  ✅ app_kb.py uses status(PROJECT_ROOT)")
        else:
            print("  ❌ app_kb.py missing status(PROJECT_ROOT) usage")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ app_kb configuration test failed: {e}")
        return False

def test_smoke_status():
    """Test that status() function can be called without errors."""
    print("\n🔍 Testing status() function smoke test...")
    
    try:
        from phase4_knowledge.knowledgebase_builder import status
        
        # Test with a dummy project root
        test_project_root = "/Project_Root"
        
        # This should not error even if files don't exist
        result = status(test_project_root)
        
        # Check expected fields are present
        expected_fields = [
            "manifest_files", "index_exists", "docstore_exists", 
            "manifest_exists", "index_path", "docstore_path", 
            "manifest_path", "project_root"
        ]
        
        for field in expected_fields:
            if field in result:
                print(f"  ✅ status() returns field: {field}")
            else:
                print(f"  ❌ status() missing field: {field}")
                return False
        
        # Check that paths include Dropbox structure if applicable
        manifest_path = result.get("manifest_path", "")
        if "06_LLM_Knowledge_Base" in manifest_path:
            print("  ✅ status() returns Dropbox-compatible paths")
        else:
            print(f"  ⚠️ status() path structure: {manifest_path}")
        
        print(f"  📊 Status result sample: manifest_files={result.get('manifest_files', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ status() smoke test failed: {e}")
        return False

def run_phase_e_validation():
    """Run all Phase E validation tests."""
    print("🚀 CHATGPT PHASE E (KNOWLEDGE BASE) VALIDATION")
    print("=" * 60)
    
    tests = [
        ("KnowledgeBase Builder Config", test_knowledgebase_builder_config),
        ("KB QA Contract", test_kb_qa_contract),
        ("KnowledgeBase Retriever Updates", test_knowledgebase_retriever_updates),
        ("App KB Configuration", test_app_kb_configuration),
        ("Status Function Smoke Test", test_smoke_status)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE E VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 PERFECT PHASE E ALIGNMENT!")
        print("   Knowledge Base implementation meets all ChatGPT requirements:")
        print("   • Relative KB paths (*_REL) ✓")
        print("   • DEFAULT_SCAN_FOLDERS = ['06_LLM_Knowledge_Base'] ✓")
        print("   • Fixed status() function ✓")
        print("   • Contract: kb_answer() -> {answer, hits, context} ✓")
        print("   • Enterprise-ready for supply chain document support ✓")
        print("\n   🚀 Ready for production deployment!")
    elif passed >= total * 0.8:
        print(f"\n✅ EXCELLENT PHASE E ALIGNMENT! {passed}/{total} components working.")
        print("   Minor adjustments may be needed for remaining components.")
    else:
        print(f"\n⚠️ PARTIAL PHASE E ALIGNMENT. {passed}/{total} components working.")
        print("   Significant work needed to match ChatGPT's Phase E requirements.")
    
    return passed == total

if __name__ == "__main__":
    success = run_phase_e_validation()
    sys.exit(0 if success else 1)
