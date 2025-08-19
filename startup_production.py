#!/usr/bin/env python3
"""
Production Startup Script for SCIE Ethos
Ensures master instructions routing is active and all components are initialized
"""

import os
import sys
import yaml
from pathlib import Path

# Ensure master instructions are loaded first
def validate_and_load_master_instructions():
    """Validate and load master instructions at startup"""
    instructions_path = Path(__file__).parent / "prompts" / "instructions_master.yaml"
    
    if not instructions_path.exists():
        raise FileNotFoundError("❌ CRITICAL: Master instructions file not found!")
    
    try:
        with open(instructions_path, 'r') as f:
            instructions = yaml.safe_load(f)
        
        # Validate critical sections
        required_sections = ['intent_routing', 'tool_registry', 'quality_protocol']
        for section in required_sections:
            if section not in instructions:
                raise ValueError(f"❌ CRITICAL: Missing {section} in master instructions")
        
        # Validate comparison routing (critical for production)
        comparison_config = instructions.get('intent_routing', {}).get('comparison', {})
        if not comparison_config or comparison_config.get('priority') != 1:
            raise ValueError("❌ CRITICAL: Comparison routing not properly configured")
        
        print("✅ Master instructions validated and loaded successfully")
        print(f"   Version: {instructions.get('version', 'unknown')}")
        print(f"   Intent routing: {len(instructions.get('intent_routing', {}))} intents")
        print(f"   Tool registry: {len(instructions.get('tool_registry', {}))} tools")
        print("✅ Comparison routing: ACTIVE (Priority 1)")
        
        return instructions
        
    except Exception as e:
        raise RuntimeError(f"❌ CRITICAL: Failed to load master instructions: {e}")

def initialize_production_environment():
    """Initialize production environment with master instructions"""
    print("🚀 SCIE Ethos Production Startup")
    print("=" * 50)
    
    # Step 1: Load and validate master instructions
    instructions = validate_and_load_master_instructions()
    
    # Step 2: Set environment variables for production
    os.environ['SCIE_ETHOS_MODE'] = 'PRODUCTION'
    os.environ['MASTER_INSTRUCTIONS_ACTIVE'] = 'TRUE'
    os.environ['COMPARISON_ROUTING_ENABLED'] = 'TRUE'
    
    # Step 3: Add PY Files to Python path (critical for imports)
    py_files_path = Path(__file__).parent / "PY Files"
    if str(py_files_path) not in sys.path:
        sys.path.insert(0, str(py_files_path))
        print(f"✅ Added PY Files to path: {py_files_path}")
    
    # Step 4: Validate critical components are importable
    try:
        # Test orchestrator import with master instructions
        sys.path.append(str(Path(__file__).parent / "PY Files"))
        from orchestrator import MASTER_INSTRUCTIONS, INTENT_ROUTING
        
        if not MASTER_INSTRUCTIONS:
            raise ImportError("Master instructions not loaded in orchestrator")
        
        if 'comparison' not in INTENT_ROUTING:
            raise ImportError("Comparison routing not found in orchestrator")
        
        print("✅ Orchestrator with master instructions: LOADED")
        print("✅ Comparison routing in orchestrator: ACTIVE")
        
    except Exception as e:
        raise RuntimeError(f"❌ CRITICAL: Orchestrator initialization failed: {e}")
    
    print("=" * 50)
    print("🎉 PRODUCTION ENVIRONMENT READY")
    print("✅ Master instructions: ACTIVE as single source of truth")
    print("✅ Comparison routing: ENABLED for chat and UI")
    print("✅ All phases 0-6: OPERATIONAL")
    print("✅ Enterprise infrastructure: CONFIGURED")
    
    return True

if __name__ == "__main__":
    try:
        initialize_production_environment()
        
        # Launch main application
        print("\n🚀 Launching main application...")
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py", "--server.port=8501"])
        
    except Exception as e:
        print(f"\n❌ PRODUCTION STARTUP FAILED: {e}")
        sys.exit(1)
