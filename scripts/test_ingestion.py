#!/usr/bin/env python3
"""
Test script for SCIE Ethos Enhanced Ingestion Pipeline
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add the PY Files directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PY Files'))

def test_environment_setup():
    """Test environment setup and directory creation."""
    print("🔍 Testing environment setup...")
    
    try:
        from run_ingestion import setup_environment
        
        # Create temporary test directory
        test_dir = tempfile.mkdtemp(prefix="scie_ethos_test_")
        original_cwd = os.getcwd()
        
        try:
            os.chdir(test_dir)
            print(f"📁 Test directory: {test_dir}")
            
            # Run setup
            setup_environment()
            
            # Check if directories were created
            expected_dirs = [
                "Project_Root/04_Data/00_Raw_Files",
                "Project_Root/04_Data/01_Cleansed_Files",
                "Project_Root/04_Data/02_EDA_Charts",
                "Project_Root/04_Data/03_Summaries",
                "Project_Root/04_Data/04_Metadata",
                "Project_Root/04_Data/05_Merged_Comparisons",
                "Project_Root/05_Exports",
                "Project_Root/06_Logs",
                "Project_Root/06_LLM_Knowledge_Base"
            ]
            
            all_exist = True
            for expected_dir in expected_dirs:
                if os.path.exists(expected_dir):
                    print(f"✅ Directory exists: {expected_dir}")
                else:
                    print(f"❌ Directory missing: {expected_dir}")
                    all_exist = False
            
            return all_exist
            
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Environment setup test failed: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation."""
    print("\n🔍 Testing sample data creation...")
    
    try:
        from run_ingestion import create_sample_data
        
        # Create temporary test directory
        test_dir = tempfile.mkdtemp(prefix="scie_ethos_test_")
        original_cwd = os.getcwd()
        
        try:
            os.chdir(test_dir)
            
            # Create required directories
            os.makedirs("Project_Root/04_Data/00_Raw_Files", exist_ok=True)
            
            # Run sample data creation
            create_sample_data()
            
            # Check if sample files were created
            sample_dir = "Project_Root/04_Data/00_Raw_Files"
            sample_files = list(Path(sample_dir).glob("*.xlsx"))
            
            if sample_files:
                print(f"✅ Sample files created: {len(sample_files)}")
                for file in sample_files:
                    print(f"  - {file.name}")
                return True
            else:
                print("❌ No sample files created")
                return False
                
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Sample data creation test failed: {e}")
        return False

def test_pipeline_import():
    """Test that the enhanced ingestion pipeline can be imported."""
    print("\n🔍 Testing pipeline import...")
    
    try:
        from phase1_ingest.enhanced_ingestion_pipeline import EnhancedIngestionPipeline, run_enhanced_ingestion
        
        print("✅ Enhanced ingestion pipeline imported successfully")
        print(f"  - EnhancedIngestionPipeline class: {EnhancedIngestionPipeline}")
        print(f"  - run_enhanced_ingestion function: {run_enhanced_ingestion}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization with configuration."""
    print("\n🔍 Testing pipeline initialization...")
    
    try:
        from phase1_ingest.enhanced_ingestion_pipeline import EnhancedIngestionPipeline
        
        # Test with default config
        pipeline = EnhancedIngestionPipeline()
        print("✅ Pipeline initialized with default config")
        
        # Check configuration loading
        if pipeline.config:
            print(f"  - Config loaded: {len(pipeline.config)} sections")
        else:
            print("  - Config is empty (using defaults)")
        
        # Check path contract loading
        if pipeline.path_contract:
            print(f"  - Path contract loaded: {len(pipeline.path_contract)} sections")
        else:
            print("  - Path contract is empty (using defaults)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_ontology_metadata_extraction():
    """Test ontology metadata extraction functionality."""
    print("\n🔍 Testing ontology metadata extraction...")
    
    try:
        from phase1_ingest.enhanced_ingestion_pipeline import EnhancedIngestionPipeline
        import pandas as pd
        
        # Create test data
        test_data = {
            'Part_Number': ['P001', 'P002'],
            'On_Hand_Qty': [100, 200],
            'Unit_Cost': [10.50, 25.75],
            'Currency': ['USD', 'USD'],
            'Plant': ['US01', 'MX01'],
            'Lead_Time_Days': [30, 45]
        }
        test_df = pd.DataFrame(test_data)
        
        # Initialize pipeline
        pipeline = EnhancedIngestionPipeline()
        
        # Test metadata extraction
        metadata = pipeline._extract_ontology_metadata(test_df, "TestSheet", "test_inventory.xlsx")
        
        # Check key metadata fields
        required_fields = [
            "erp", "country", "income_stream", "sheet_type",
            "currency", "uom", "has_aging", "has_wip"
        ]
        
        all_fields_present = True
        for field in required_fields:
            if field in metadata:
                print(f"  ✅ {field}: {metadata[field]}")
            else:
                print(f"  ❌ {field}: missing")
                all_fields_present = False
        
        return all_fields_present
        
    except Exception as e:
        print(f"❌ Ontology metadata extraction test failed: {e}")
        return False

def test_summary_card_generation():
    """Test summary card generation functionality."""
    print("\n🔍 Testing summary card generation...")
    
    try:
        from phase1_ingest.enhanced_ingestion_pipeline import EnhancedIngestionPipeline
        import pandas as pd
        
        # Create test data
        test_data = {
            'Part_Number': ['P001'],
            'On_Hand_Qty': [100]
        }
        test_df = pd.DataFrame(test_data)
        
        # Initialize pipeline
        pipeline = EnhancedIngestionPipeline()
        
        # Create test metadata
        test_metadata = {
            "sheet_type": "inventory",
            "country": "US",
            "erp": "SAP",
            "row_count": 1,
            "col_count": 2,
            "has_aging": True,
            "lt_mean": 30.0,
            "lt_std": 5.0
        }
        
        # Test summary card generation
        summary_card = pipeline._generate_summary_card(test_metadata, test_df)
        
        if summary_card and "#" in summary_card:
            print("✅ Summary card generated successfully")
            print(f"  - Length: {len(summary_card)} characters")
            print(f"  - Contains markdown: {'#' in summary_card}")
            return True
        else:
            print("❌ Summary card generation failed")
            return False
        
    except Exception as e:
        print(f"❌ Summary card generation test failed: {e}")
        return False

def test_artifact_emission():
    """Test artifact emission functionality."""
    print("\n🔍 Testing artifact emission...")
    
    try:
        from phase1_ingest.enhanced_ingestion_pipeline import EnhancedIngestionPipeline
        
        # Create temporary test directory
        test_dir = tempfile.mkdtemp(prefix="scie_ethos_test_")
        original_cwd = os.getcwd()
        
        try:
            os.chdir(test_dir)
            
            # Create required directories
            os.makedirs("Project_Root/04_Data/04_Metadata", exist_ok=True)
            
            # Initialize pipeline
            pipeline = EnhancedIngestionPipeline()
            
            # Add some test data
            pipeline.master_catalog = [
                {"filename": "test.xlsx", "sheet_name": "Sheet1", "row_count": 100}
            ]
            pipeline.eda_profiles = {
                "test.xlsx_Sheet1": {"metadata": {"rows": 100, "cols": 5}}
            }
            pipeline.summary_cards = ["# Test Summary"]
            
            # Test artifact emission
            artifacts = pipeline.emit_artifacts()
            
            if artifacts:
                print("✅ Artifacts emitted successfully")
                for artifact_type, path in artifacts.items():
                    if os.path.exists(path):
                        print(f"  - {artifact_type}: {path} (exists)")
                    else:
                        print(f"  - {artifact_type}: {path} (missing)")
                return True
            else:
                print("❌ No artifacts emitted")
                return False
                
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(test_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Artifact emission test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 SCIE Ethos Enhanced Ingestion Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Sample Data Creation", test_sample_data_creation),
        ("Pipeline Import", test_pipeline_import),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Ontology Metadata Extraction", test_ontology_metadata_extraction),
        ("Summary Card Generation", test_summary_card_generation),
        ("Artifact Emission", test_artifact_emission)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:35} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Enhanced ingestion pipeline is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
