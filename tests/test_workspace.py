#!/usr/bin/env python
"""Workspaces, retention scan and safe deletion.

Run:  python -m pytest tests/test_workspace.py -q
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workspace


def _run(ws: Path, day: str, name: str, ok: bool, age_days: float):
    d = ws / day / name
    d.mkdir(parents=True)
    f = d / ("model_trainer_report.json" if ok else "agent_execution.log")
    f.write_text("{}")
    t = time.time() - age_days * 86400
    for p in (f, d):
        os.utime(p, (t, t))
    return d


def test_layout_and_names(tmp_path):
    assert workspace.ws_dir(str(tmp_path), "default") == tmp_path
    assert workspace.ws_dir(str(tmp_path), "ngoc") == tmp_path / "workspaces" / "ngoc"
    workspace.create(str(tmp_path), "ngoc")
    assert workspace.list_workspaces(str(tmp_path)) == ["default", "ngoc"]
    for bad in ("", "../x", "a/b", "workspaces", "x" * 80):
        with pytest.raises(ValueError):
            workspace.ws_dir(str(tmp_path), bad) if bad else (_ for _ in ()).throw(ValueError())


def test_scan_classifies_failed_old_kept_and_recent(tmp_path):
    ws = workspace.create(str(tmp_path), "w")
    failed = _run(ws, "2026-10-01", "run_01", ok=False, age_days=1)
    fresh_fail = _run(ws, "2026-10-01", "run_02", ok=False, age_days=0.01)   # may still be running
    old = _run(ws, "2026-10-01", "run_03", ok=True, age_days=10)
    kept = _run(ws, "2026-10-01", "run_04", ok=True, age_days=10)
    (kept / workspace.KEEP_MARKER).touch()
    recent = _run(ws, "2026-10-20", "run_01", ok=True, age_days=1)
    up = workspace.uploads_dir(str(tmp_path), "w") / "old.csv"
    up.write_text("a\n1")
    os.utime(up, (time.time() - 9 * 86400,) * 2)

    rep = workspace.scan(ws, 7)
    assert [i["path"] for i in rep["failed_runs"]] == [str(failed)]
    assert [i["path"] for i in rep["old_runs"]] == [str(old)]
    assert [i["path"] for i in rep["old_uploads"]] == [str(up)]
    # a running job's dir is never reported
    assert workspace.scan(ws, 7, active_run_dirs=[str(failed)])["failed_runs"] == []
    assert fresh_fail.exists() and recent.exists()


def test_delete_only_inside_workspace(tmp_path):
    ws = workspace.create(str(tmp_path), "w")
    run = _run(ws, "2026-10-01", "run_01", ok=True, age_days=10)
    outside = tmp_path / "precious.csv"
    outside.write_text("x")
    other_ws = workspace.create(str(tmp_path), "other")
    other_run = _run(other_ws, "2026-10-01", "run_01", ok=True, age_days=10)
    r = workspace.delete(ws, [str(run), str(outside), str(other_run), str(ws), str(ws / "2026-10-01")])
    assert r["deleted"] == 1
    assert not run.exists() and outside.exists() and other_run.exists() and ws.exists()
    assert not (ws / "2026-10-01").exists()                 # empty date folder removed


def test_keep_marks_runs(tmp_path):
    ws = workspace.create(str(tmp_path), "w")
    old = _run(ws, "2026-10-01", "run_01", ok=True, age_days=10)
    assert workspace.keep([str(old)]) == 1
    assert workspace.scan(ws, 7)["old_runs"] == []
