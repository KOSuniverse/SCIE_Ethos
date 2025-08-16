#!/usr/bin/env python3
import os, shutil, sys
def main(base="Project_Root/06_LLM_Knowledge_Base"):
    faiss = os.path.join(base, "document_index.faiss")
    archive = os.path.join(base, "_archive")
    if not os.path.exists(faiss):
        print("No FAISS index found (nothing to do).")
        return 0
    os.makedirs(archive, exist_ok=True)
    dest = os.path.join(archive, "document_index.faiss")
    shutil.move(faiss, dest)
    print(f"Moved FAISS index to: {dest}")
    return 0
if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
