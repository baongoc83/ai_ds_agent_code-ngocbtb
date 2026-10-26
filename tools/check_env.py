#!/usr/bin/env python
"""Verify the active interpreter matches the reference environment.

    python tools/check_env.py            # report, exit 1 on any mismatch
    python tools/check_env.py --dev      # also check requirements-dev.txt

Checks the Python minor version (.python-version) and every pin in
requirements-lock.txt — the same versions final_model.pkl / feature specs were
pickled with. Run it after setup_env.* and whenever runs behave differently on
another machine.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _pins(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line and not line.startswith("-"):
            name, ver = line.split("==", 1)
            out[name.strip()] = ver.split(";")[0].strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    problems = []

    want_py = (REPO / ".python-version").read_text().strip() if (REPO / ".python-version").exists() else None
    have_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    if want_py and not have_py.startswith(want_py):
        problems.append(f"python: need {want_py}, running {have_py} ({sys.executable})")

    files = ["requirements-lock.txt"] + (["requirements-dev.txt"] if args.dev else [])
    for f in files:
        for name, ver in _pins(REPO / f).items():
            try:
                got = md.version(name)
            except md.PackageNotFoundError:
                problems.append(f"{name}: missing (need {ver})")
                continue
            if got != ver:
                problems.append(f"{name}: need {ver}, installed {got}")

    if problems:
        print(f"Environment MISMATCH ({len(problems)}):")
        for p in problems:
            print("  -", p)
        print("\nFix: run setup_env.ps1 / setup_env.sh, or  pip install -r requirements-lock.txt")
        return 1
    print(f"Environment OK — Python {have_py}, {sum(len(_pins(REPO / f)) for f in files)} pinned packages match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
