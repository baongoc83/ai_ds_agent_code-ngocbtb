"""Background pipeline runner — one OS process per job.

The pipeline is long-running (minutes to hours) and Config is a process-global
class (RUN_DIR, S3 credentials, auto-config knobs). Running jobs in threads of
the Streamlit process made two users' runs overwrite each other, so every job
now runs in its own process (ui/job_worker.py) with that session's settings
passed through the environment. The Streamlit process only polls files:

  <JOB_ROOT>/<job_id>/params.json   pipeline.run arguments
  <JOB_ROOT>/<job_id>/status.json   starting | running | done | error (+ run_dir, result)
  <JOB_ROOT>/<job_id>/output.log    worker stdout (live agent log)

Concurrency is capped by UI_MAX_CONCURRENT_JOBS (default 2): each job is CPU /
RAM heavy, so unlimited parallel runs would only make all of them slower.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
JOB_ROOT = Path(os.getenv("UI_JOB_DIR", Path(tempfile.gettempdir()) / "automl_ui_jobs"))
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("UI_MAX_CONCURRENT_JOBS", 2)))

# job_id -> Popen, for jobs started by THIS Streamlit process (status itself is on disk)
_PROCS: Dict[str, subprocess.Popen] = {}
_LOCK = threading.Lock()


def _tail(path: Path, n: int = 8) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    return "".join(lines[-n:]).strip()


def _alive(job_id: str) -> bool:
    proc = _PROCS.get(job_id)
    return proc is not None and proc.poll() is None


def running_count() -> int:
    """Jobs currently running in this server (all sessions)."""
    with _LOCK:
        return sum(1 for jid in _PROCS if _alive(jid))


def is_running(job_ids: Optional[Iterable[str]] = None) -> bool:
    """True when any of `job_ids` (default: any job of this server) is running."""
    with _LOCK:
        ids = list(_PROCS) if job_ids is None else [j for j in job_ids if j]
        return any(_alive(j) for j in ids)


def active_run_dirs() -> list:
    """Run dirs of jobs still running in this server — never offered for clean-up."""
    with _LOCK:
        ids = [j for j in _PROCS if _alive(j)]
    out = []
    for j in ids:
        st = get_status(j) or {}
        if st.get("run_dir"):
            out.append(st["run_dir"])
    return out


def can_submit() -> bool:
    return running_count() < MAX_CONCURRENT_JOBS


def submit_job(run_params: dict, env_overrides: Optional[Dict[str, str]] = None) -> str:
    """Start the pipeline in its own process. Returns the job id.

    env_overrides: session-scoped settings (pinned knobs, S3 credentials). They
    go only into the child's environment — never into this process or a file.
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "params.json").write_text(json.dumps(run_params, default=str), encoding="utf-8")
    # Overrides travel as ONE env var applied by the worker after config.py loaded
    # .env (load_dotenv(override=True) would otherwise replace them).
    overrides = {k: str(v) for k, v in (env_overrides or {}).items() if v not in (None, "")}
    env = {**os.environ, "AUTOML_JOB_OVERRIDES": json.dumps(overrides),
           "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    log = open(job_dir / "output.log", "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen([sys.executable, "-m", "ui.job_worker", str(job_dir)],
                            cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=flags)
    log.close()                       # the child holds its own handle
    with _LOCK:
        _PROCS[job_id] = proc
    return job_id


def get_status(job_id: str) -> Optional[dict]:
    """{status, run_dir, result, error, traceback, log} or None for an unknown job."""
    job_dir = JOB_ROOT / job_id
    if not job_dir.exists():
        return None
    try:
        st = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {"status": "starting"}
    with _LOCK:
        proc = _PROCS.get(job_id)
        exited = proc is not None and proc.poll() is not None
    if st.get("status") in ("starting", "running"):
        if exited:
            # the worker died without writing a final state (OOM kill, crash)
            st = {**st, "status": "error",
                  "error": f"Tiến trình train dừng bất thường (exit code {proc.returncode}).",
                  "traceback": _tail(job_dir / "output.log", 40)}
        elif proc is None:
            # started by an earlier Streamlit process (server restarted) — trust the file
            pass
        st["status"] = "running" if st["status"] == "starting" else st["status"]
    st["log"] = _tail(job_dir / "output.log")
    return st


def cancel_job(job_id: str) -> bool:
    with _LOCK:
        proc = _PROCS.get(job_id)
    if proc is None or proc.poll() is not None:
        return False
    proc.terminate()
    return True


def refresh_log(job_id: str) -> str:
    """Back-compat: the log tail is part of get_status() now."""
    st = get_status(job_id) or {}
    return st.get("log", "")


def run_dir_for(job_id: str) -> Optional[str]:
    st = get_status(job_id)
    return st.get("run_dir") if st else None


def list_successful_runs(output_dir: str) -> list:
    """Return sorted (date, run_name, run_dir) tuples for runs that completed.

    A run counts as successful when it produced at least one of the final
    report artifacts (feature_engineer_report.json / model_trainer_report.json).
    Sorted newest first so the UI can default to the latest run.
    """
    base = Path(output_dir)
    if not base.exists():
        return []
    found = []
    for date_dir in sorted(base.iterdir(), reverse=True):
        if not date_dir.is_dir() or not date_dir.name[:4].isdigit():
            continue
        for run_dir in sorted(date_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            if (run_dir / "feature_engineer_report.json").exists() or (
                run_dir / "model_trainer_report.json"
            ).exists():
                found.append((date_dir.name, run_dir.name, str(run_dir)))
    return found
