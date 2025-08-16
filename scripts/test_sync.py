#!/usr/bin/env python3
"""
Test script for SCIE Ethos Dropbox sync functionality
"""

import os
import sys
import json
from pathlib import Path

def test_environment():
    """Test that required environment variables are set."""
    print("🔍 Testing environment...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "DROPBOX_APP_KEY", 
        "DROPBOX_APP_SECRET",
        "DROPBOX_REFRESH_TOKEN"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing required environment variables: {missing}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_dropbox_connection():
    """Test Dropbox connection."""
    print("\n🔍 Testing Dropbox connection...")
    
    try:
        from dropbox_sync import init_dropbox
        dbx = init_dropbox()
        
        # Try to list root folder
        resp = dbx.files_list_folder("", recursive=False)
        print(f"✅ Dropbox connection successful (found {len(resp.entries)} root items)")
        return True
        
    except Exception as e:
        print(f"❌ Dropbox connection failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI connection."""
    print("\n🔍 Testing OpenAI connection...")
    
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Try to list assistants
        assistants = client.beta.assistants.list(limit=5)
        print(f"✅ OpenAI connection successful (found {len(assistants.data)} assistants)")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def test_assistant_creation():
    """Test assistant creation."""
    print("\n🔍 Testing assistant creation...")
    
    try:
        from dropbox_sync import create_assistant
        assistant_id = create_assistant()
        print(f"✅ Assistant created successfully: {assistant_id}")
        return True
        
    except Exception as e:
        print(f"❌ Assistant creation failed: {e}")
        return False

def test_sync_functionality():
    """Test basic sync functionality."""
    print("\n🔍 Testing sync functionality...")
    
    try:
        from dropbox_sync import sync_dropbox_to_assistant
        
        # Run a small sync test
        print("🔄 Running test sync...")
        sync_dropbox_to_assistant(batch_size=2)
        print("✅ Sync test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Sync test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 SCIE Ethos Sync Test Suite")
    print("=" * 40)
    
    tests = [
        ("Environment", test_environment),
        ("Dropbox Connection", test_dropbox_connection),
        ("OpenAI Connection", test_openai_connection),
        ("Assistant Creation", test_assistant_creation),
        ("Sync Functionality", test_sync_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Sync functionality is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
