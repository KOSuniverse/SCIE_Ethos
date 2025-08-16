#!/usr/bin/env python3
"""
CLI script to run the SCIE Ethos Enhanced Ingestion Pipeline
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the PY Files directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PY Files'))

def setup_environment():
    """Setup environment variables and paths."""
    # Set default paths
    os.environ.setdefault("PROJECT_ROOT", "Project_Root")
    
    # Create necessary directories
    base_dirs = [
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
    
    for base_dir in base_dirs:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        print(f"📁 Ensured directory: {base_dir}")

def create_sample_data():
    """Create sample data files for testing if none exist."""
    sample_dir = "Project_Root/04_Data/00_Raw_Files"
    
    if not any(Path(sample_dir).glob("*.xlsx")):
        print("📊 Creating sample data files for testing...")
        
        try:
            import pandas as pd
            
            # Sample inventory data
            inventory_data = {
                'Part_Number': ['P001', 'P002', 'P003', 'P004', 'P005'],
                'Description': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
                'On_Hand_Qty': [100, 250, 75, 300, 150],
                'Unit_Cost': [10.50, 25.75, 15.25, 8.99, 45.00],
                'Currency': ['USD', 'USD', 'USD', 'USD', 'USD'],
                'UOM': ['PCS', 'PCS', 'PCS', 'PCS', 'PCS'],
                'Plant': ['US01', 'US01', 'MX01', 'US01', 'CA01'],
                'Country': ['US', 'US', 'MX', 'US', 'CA'],
                'Aging_0_30': [50, 100, 25, 150, 75],
                'Aging_31_60': [30, 100, 25, 100, 50],
                'Aging_61_90': [15, 35, 15, 35, 15],
                'Aging_90_plus': [5, 15, 10, 15, 10]
            }
            
            inventory_df = pd.DataFrame(inventory_data)
            inventory_path = os.path.join(sample_dir, "sample_inventory.xlsx")
            inventory_df.to_excel(inventory_path, index=False, sheet_name="Inventory")
            print(f"✅ Created: {inventory_path}")
            
            # Sample WIP data
            wip_data = {
                'Job_ID': ['J001', 'J002', 'J003', 'J004'],
                'Part_Number': ['P001', 'P002', 'P003', 'P004'],
                'Work_Center': ['WC01', 'WC02', 'WC01', 'WC03'],
                'Qty_In_Process': [25, 50, 15, 40],
                'Start_Date': ['2024-01-15', '2024-01-20', '2024-01-25', '2024-01-30'],
                'Due_Date': ['2024-02-15', '2024-02-20', '2024-02-25', '2024-03-01'],
                'Lead_Time_Days': [31, 31, 31, 30],
                'Plant': ['US01', 'US01', 'MX01', 'US01']
            }
            
            wip_df = pd.DataFrame(wip_data)
            wip_path = os.path.join(sample_dir, "sample_wip.xlsx")
            wip_df.to_excel(wip_path, index=False, sheet_name="WIP")
            print(f"✅ Created: {wip_path}")
            
        except ImportError:
            print("⚠️ pandas not available - skipping sample data creation")
        except Exception as e:
            print(f"⚠️ Failed to create sample data: {e}")

def run_ingestion(file_paths, config_path, knowledge_base_path, verbose=False):
    """Run the enhanced ingestion pipeline."""
    try:
        from phase1_ingest.enhanced_ingestion_pipeline import run_enhanced_ingestion
        
        def reporter(event: str, payload: dict):
            if verbose:
                print(f"📊 {event}: {payload}")
            else:
                # Show only important events
                if event in ["pipeline_start", "pipeline_complete", "file_error"]:
                    print(f"📊 {event}: {payload}")
        
        print(f"🧠 Starting Enhanced Ingestion Pipeline...")
        print(f"📁 Files to process: {len(file_paths)}")
        print(f"⚙️  Config: {config_path}")
        if knowledge_base_path:
            print(f"📚 Knowledge Base: {knowledge_base_path}")
        
        results = run_enhanced_ingestion(
            file_paths=file_paths,
            config_path=config_path,
            reporter=reporter
        )
        
        return results
        
    except ImportError as e:
        print(f"❌ Failed to import ingestion pipeline: {e}")
        print("💡 Make sure you're running from the project root directory")
        return None
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return None

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SCIE Ethos Enhanced Ingestion Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific files
  python scripts/run_ingestion.py data1.xlsx data2.xlsx
  
  # Process all Excel files in raw data directory
  python scripts/run_ingestion.py --raw-data-dir
  
  # Use custom config
  python scripts/run_ingestion.py --config custom_rules.yaml data.xlsx
  
  # Include knowledge base ingestion
  python scripts/run_ingestion.py --kb data.xlsx
  
  # Verbose output
  python scripts/run_ingestion.py --verbose data.xlsx
        """
    )
    
    parser.add_argument(
        "files",
        nargs="*",
        help="Excel files to process"
    )
    
    parser.add_argument(
        "--raw-data-dir",
        action="store_true",
        help="Process all Excel files in Project_Root/04_Data/00_Raw_Files"
    )
    
    parser.add_argument(
        "--config",
        default="prompts/ingest_rules.yaml",
        help="Path to ingestion rules configuration (default: prompts/ingest_rules.yaml)"
    )
    
    parser.add_argument(
        "--kb",
        action="store_true",
        help="Include knowledge base ingestion"
    )
    
    parser.add_argument(
        "--kb-path",
        help="Custom knowledge base path"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup environment and create sample data"
    )
    
    args = parser.parse_args()
    
    # Setup environment if requested
    if args.setup:
        setup_environment()
        create_sample_data()
        print("✅ Environment setup complete!")
        return
    
    # Determine files to process
    file_paths = []
    
    if args.raw_data_dir:
        raw_dir = "Project_Root/04_Data/00_Raw_Files"
        if os.path.exists(raw_dir):
            excel_files = list(Path(raw_dir).glob("*.xlsx")) + list(Path(raw_dir).glob("*.xls"))
            file_paths = [str(f) for f in excel_files]
            print(f"📁 Found {len(file_paths)} Excel files in {raw_dir}")
        else:
            print(f"❌ Raw data directory not found: {raw_dir}")
            return
    elif args.files:
        file_paths = args.files
    else:
        print("❌ No files specified. Use --raw-data-dir or provide file paths.")
        print("💡 Use --help for usage information.")
        return
    
    # Validate files exist
    existing_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"⚠️ File not found: {file_path}")
    
    if not existing_files:
        print("❌ No valid files to process.")
        return
    
    # Validate config exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    # Run ingestion
    knowledge_base_path = args.kb_path if args.kb_path else ("Project_Root/06_LLM_Knowledge_Base" if args.kb else None)
    
    results = run_ingestion(
        file_paths=existing_files,
        config_path=args.config,
        knowledge_base_path=knowledge_base_path,
        verbose=args.verbose
    )
    
    if results:
        print("\n" + "="*60)
        print("📋 INGESTION RESULTS SUMMARY")
        print("="*60)
        print(f"Files processed: {results['files_processed']}")
        print(f"Files successful: {results['files_successful']}")
        print(f"Files failed: {results['files_failed']}")
        print(f"Total sheets: {results['total_sheets']}")
        
        if results['artifacts_emitted']:
            print(f"\n✅ Artifacts emitted:")
            for artifact_type, path in results['artifacts_emitted'].items():
                print(f"  - {artifact_type}: {path}")
        
        if results['knowledge_base_ingested']:
            kb_results = results['knowledge_base_ingested']
            print(f"\n📚 Knowledge Base:")
            print(f"  - Documents ingested: {kb_results.get('ingested', 0)}")
            if kb_results.get('errors'):
                print(f"  - Errors: {len(kb_results['errors'])}")
        
        if results['errors']:
            print(f"\n❌ Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
            if len(results['errors']) > 3:
                print(f"  ... and {len(results['errors']) - 3} more")
        
        print("\n🎉 Enhanced ingestion pipeline complete!")
        
        # Exit with error code if any files failed
        if results['files_failed'] > 0:
            sys.exit(1)
    else:
        print("❌ Ingestion pipeline failed to run.")
        sys.exit(1)

if __name__ == "__main__":
    main()
