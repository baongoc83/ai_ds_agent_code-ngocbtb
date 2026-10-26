"""One UI training job = one OS process.

    python -m ui.job_worker <job_dir>

Config is a process-global class: two jobs in the same Streamlit process would
overwrite each other's RUN_DIR, S3 credentials and auto-config knobs. Running
each job here keeps them isolated. The parent (ui/runner.py) passes:

  <job_dir>/params.json   pipeline.run(**params) arguments (no secrets)
  environment             session settings — pinned knobs (YAML editor) and
                          S3 credentials — set BEFORE config is imported, so
                          they are plain env values to this process and never
                          touch the disk.

Progress goes to stdout (AgentLogger prints every entry), which the parent
redirects to <job_dir>/output.log. State is written to <job_dir>/status.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _write_status(job_dir: Path, **state) -> None:
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(state, default=str, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, job_dir / "status.json")          # atomic for the polling reader


def _jsonable(metrics: dict) -> dict:
    out = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        out[k] = v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v)
    return out


def _apply_overrides(overrides: dict) -> None:
    """Session settings win over .env: config.py runs load_dotenv(override=True)
    at import, which would silently replace plain env vars with the .env values
    (e.g. the server's S3 keys or OUTPUT_DIR). So they arrive in one JSON env var
    and are applied AFTER Config is imported — to os.environ (read again by
    Config.reset_auto_config) and to the live Config attributes."""
    from config import Config
    for key, value in overrides.items():
        os.environ[key] = str(value)
        if not hasattr(Config, key):
            continue
        cur = getattr(Config, key)
        if isinstance(cur, bool):
            cast = str(value).lower() == "true"
        elif isinstance(cur, int):
            cast = int(float(value))
        elif isinstance(cur, float):
            cast = float(value)
        else:
            cast = str(value)
        if key == "OUTPUT_DIR" and not os.path.isabs(cast):
            cast = str(REPO / cast)
        setattr(Config, key, cast)


def main(job_dir: str) -> int:
    jd = Path(job_dir)
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    params = json.loads((jd / "params.json").read_text(encoding="utf-8"))
    overrides = json.loads(os.environ.pop("AUTOML_JOB_OVERRIDES", "{}") or "{}")
    _write_status(jd, status="starting", pid=os.getpid())
    try:
        _apply_overrides(overrides)
        from config import Config
        from pipeline import AutoMLPipeline

        pipeline = AutoMLPipeline()                    # allocates this job's RUN_DIR
        _write_status(jd, status="running", pid=os.getpid(), run_dir=Config.RUN_DIR)
        result = pipeline.run(**params)
        _write_status(jd, status="done", pid=os.getpid(), run_dir=Config.RUN_DIR,
                      result=_jsonable(result))
        return 0
    except BaseException as e:  # noqa: BLE001 — report every failure to the UI
        run_dir = None
        try:
            from config import Config
            run_dir = Config.RUN_DIR
        except Exception:
            pass
        _write_status(jd, status="error", pid=os.getpid(), run_dir=run_dir,
                      error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
