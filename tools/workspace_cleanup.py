#!/usr/bin/env python
"""Workspace clean-up from the command line (same rules as the UI panel).

    python tools/workspace_cleanup.py                       # report for every workspace
    python tools/workspace_cleanup.py --workspace ngoc      # one workspace
    python tools/workspace_cleanup.py --workspace ngoc --delete failed
    python tools/workspace_cleanup.py --workspace ngoc --delete old --days 14
    python tools/workspace_cleanup.py --workspace ngoc --keep old   # mark old runs as kept

Deleting always asks for confirmation (type "yes"); --yes skips it for cron jobs.
Only <date>/run_NN folders and _uploads files inside the workspace are ever removed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workspace  # noqa: E402
from config import Config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=None, help="workspace name (default: all)")
    ap.add_argument("--days", type=int, default=workspace.RETENTION_DAYS)
    ap.add_argument("--delete", choices=["failed", "old", "all"])
    ap.add_argument("--keep", choices=["old"])
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    names = [args.workspace] if args.workspace else workspace.list_workspaces(Config.OUTPUT_DIR)
    for name in names:
        ws = workspace.ws_dir(Config.OUTPUT_DIR, name)
        rep = workspace.scan(ws, args.days)
        print(f"\n== workspace '{name}'  ({ws})")
        for kind, label in (("failed_runs", "failed / interrupted runs"),
                            ("old_runs", f"runs older than {args.days} days"),
                            ("old_uploads", f"uploads older than {args.days} days")):
            items = rep[kind]
            print(f"  {label}: {len(items)} ({sum(i['size_mb'] for i in items):,.0f} MB)")
            for i in items[:20]:
                print(f"     - {i['name']:<40} {i['age_days']:>6} d {i['size_mb']:>9,.1f} MB")
        if args.keep == "old":
            print(f"  kept {workspace.keep(i['path'] for i in rep['old_runs'])} run(s)")
        if not args.delete:
            continue
        targets = (rep["failed_runs"] if args.delete in ("failed", "all") else []) + \
                  (rep["old_runs"] + rep["old_uploads"] if args.delete in ("old", "all") else [])
        if not targets:
            continue
        if not args.yes and input(f"  Delete {len(targets)} item(s) from '{name}'? type 'yes': ").strip() != "yes":
            print("  skipped")
            continue
        r = workspace.delete(ws, [i["path"] for i in targets])
        print(f"  deleted {r['deleted']} item(s), freed {r['freed_mb']:,.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
