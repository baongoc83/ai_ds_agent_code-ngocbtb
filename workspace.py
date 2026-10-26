"""Per-user workspaces + retention / clean-up.

Layout (OUTPUT_DIR = Config.OUTPUT_DIR, e.g. ./outputs):

    OUTPUT_DIR/                       workspace "default" (also every pre-workspace run)
        2026-10-25/run_01/ ...
        _uploads/                     files uploaded through the UI
    OUTPUT_DIR/workspaces/<name>/     any other workspace — same layout

A run is a directory <YYYY-MM-DD>/run_NN. It is:
  * successful  — model_trainer_report.json or feature_engineer_report.json exists
  * failed      — neither exists AND it is not running (no recent writes)
  * kept        — contains a `.keep` marker: never reported as old

Nothing here deletes on its own: `scan()` reports, the UI / CLI asks the user,
`delete()` removes only paths that live inside the workspace.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT = "default"
RETENTION_DAYS = 7
UPLOADS = "_uploads"
KEEP_MARKER = ".keep"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# a run dir written to within this window may still be training in another process
_ACTIVE_WINDOW_S = 6 * 3600


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or "")) and name != "workspaces"


def ws_dir(output_dir: str, name: str) -> Path:
    if not name or name == DEFAULT:
        return Path(output_dir)
    if not valid_name(name):
        raise ValueError(f"Tên workspace không hợp lệ: {name!r} (chỉ chữ, số, _ . -)")
    return Path(output_dir) / "workspaces" / name


def list_workspaces(output_dir: str) -> List[str]:
    root = Path(output_dir) / "workspaces"
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and valid_name(p.name)) if root.exists() else []
    return [DEFAULT] + names


def create(output_dir: str, name: str) -> Path:
    d = ws_dir(output_dir, name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def uploads_dir(output_dir: str, name: str) -> Path:
    d = ws_dir(output_dir, name) / UPLOADS
    d.mkdir(parents=True, exist_ok=True)
    return d


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _last_write(p: Path) -> float:
    times = [f.stat().st_mtime for f in p.rglob("*") if f.is_file()] if p.is_dir() else []
    return max(times + [p.stat().st_mtime])


def iter_runs(ws: Path) -> Iterable[Path]:
    if not ws.exists():
        return
    for date_dir in sorted(ws.iterdir()):
        if date_dir.is_dir() and date_dir.name[:4].isdigit():
            for run in sorted(date_dir.iterdir()):
                if run.is_dir() and run.name.startswith("run_"):
                    yield run


def is_successful(run: Path) -> bool:
    return (run / "model_trainer_report.json").exists() or (run / "feature_engineer_report.json").exists()


def scan(ws: Path, retention_days: int = RETENTION_DAYS,
         active_run_dirs: Optional[Iterable[str]] = None, now: Optional[float] = None) -> Dict[str, List[dict]]:
    """Report clean-up candidates (never deletes).

    failed_runs  : not successful, not running, no write for 6 h
    old_runs     : successful runs older than `retention_days` without a .keep marker
    old_uploads  : files in _uploads older than `retention_days`
    Each item: {path, name, age_days, size_mb}.
    """
    now = now or time.time()
    active = {str(Path(p).resolve()) for p in (active_run_dirs or []) if p}
    out: Dict[str, List[dict]] = {"failed_runs": [], "old_runs": [], "old_uploads": []}

    def item(p: Path, last: float) -> dict:
        return {"path": str(p), "name": f"{p.parent.name}/{p.name}", "age_days": round((now - last) / 86400, 1),
                "size_mb": round(_size(p) / 1e6, 1)}

    for run in iter_runs(ws):
        if str(run.resolve()) in active:
            continue
        last = _last_write(run)
        if not is_successful(run):
            if now - last > _ACTIVE_WINDOW_S:
                out["failed_runs"].append(item(run, last))
        elif not (run / KEEP_MARKER).exists() and now - last > retention_days * 86400:
            out["old_runs"].append(item(run, last))
    up = ws / UPLOADS
    if up.exists():
        for f in up.iterdir():
            if f.is_file() and now - f.stat().st_mtime > retention_days * 86400:
                out["old_uploads"].append({"path": str(f), "name": f.name,
                                           "age_days": round((now - f.stat().st_mtime) / 86400, 1),
                                           "size_mb": round(f.stat().st_size / 1e6, 1)})
    return out


def _inside(ws: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(ws.resolve())
        return p.resolve() != ws.resolve()
    except ValueError:
        return False


def delete(ws: Path, paths: Iterable[str]) -> Dict[str, int]:
    """Delete runs / uploads — only paths strictly inside this workspace, and for
    run dirs only <date>/run_NN folders (never a date folder or the workspace)."""
    n, freed = 0, 0
    for s in paths:
        p = Path(s)
        if not p.exists() or not _inside(ws, p):
            continue
        is_run = p.is_dir() and p.name.startswith("run_") and p.parent.name[:4].isdigit()
        is_upload = p.is_file() and p.parent.name == UPLOADS
        if not (is_run or is_upload):
            continue
        freed += _size(p)
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        n += 1
        if is_run and p.parent.exists() and not any(p.parent.iterdir()):
            p.parent.rmdir()                       # empty date folder
    return {"deleted": n, "freed_mb": round(freed / 1e6, 1)}


def keep(paths: Iterable[str]) -> int:
    """Mark runs as 'keep' so retention never proposes them again."""
    n = 0
    for s in paths:
        p = Path(s)
        if p.is_dir() and p.name.startswith("run_"):
            (p / KEEP_MARKER).touch()
            n += 1
    return n
