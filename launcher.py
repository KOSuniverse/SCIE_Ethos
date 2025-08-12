#!/usr/bin/env python3
# launcher.py — SCIE Ethos Interface Launcher

import sys
import subprocess
import os
from pathlib import Path

def main():
    print("🧠 SCIE Ethos LLM Assistant Launcher")
    print("=" * 50)
    print()
    print("Available Interfaces:")
    print("1. 💬 Chat Assistant (Architecture-Compliant)")
    print("2. 🔧 Data Processing Workflows")
    print("3. 🛠️ Admin & Sync Tools")
    print()
    
    while True:
        choice = input("Select interface (1-3, or 'q' to quit): ").strip().lower()
        
        if choice == 'q' or choice == 'quit':
            print("👋 Goodbye!")
            break
        
        elif choice == '1':
            print("\n🚀 Launching Chat Assistant...")
            print("Running: streamlit run chat_ui.py")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "chat_ui.py"])
            break
        
        elif choice == '2':
            print("\n🔧 Launching Data Processing Interface...")
            print("Running: streamlit run main.py")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])
            break
        
        elif choice == '3':
            print("\n🛠️ Launching Admin & Sync Tools...")
            print("Running: streamlit run pages/01_Admin_Sync.py")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "pages/01_Admin_Sync.py"])
            break
        
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 'q'.")
            print()

if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have Streamlit installed: pip install streamlit")
