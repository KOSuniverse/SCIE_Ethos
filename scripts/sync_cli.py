# scripts/sync_cli.py
import argparse
from dropbox_sync import sync_dropbox_to_assistant

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Dropbox files to Assistant")
    parser.add_argument("--dry-run", action="store_true", help="List files but don't upload")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run not yet implemented — use dropbox_sync directly for now.")
    else:
        sync_dropbox_to_assistant()
