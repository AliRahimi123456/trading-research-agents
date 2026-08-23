import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.config import DATA_DIR

_HERE = Path(__file__).resolve().parent
ENGINE_FILE = _HERE / "engine.py"
RUNNER_FILE = _HERE / "runner.py"


def _scrubbed_env(sandbox: Path) -> dict:
    """Build the child process's environment from an allowlist, not the parent's copy minus secrets."""
    allowlist = ["PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
                 "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"]
    env = {k: os.environ[k] for k in allowlist if k in os.environ}
    env.update({
        "HOME": str(sandbox), "USERPROFILE": str(sandbox),
        "TEMP": str(sandbox), "TMP": str(sandbox), "TMPDIR": str(sandbox),
        "PYTHONHASHSEED": "1", "PYTHONUTF8": "1",
    })
    return env


def run_isolated(strategy_path, params: dict, split: str, cost_bps: float = 10.0, timeout: int = 120) -> dict:
    """Run one strategy, on one split, in its own throwaway process with no access
    to this project's credentials, source tree, or private/holdout data."""
    sandbox = Path(tempfile.mkdtemp(prefix="strat_"))
    (sandbox / "data").mkdir()
    try:
        for field in ["adj_close", "close", "volume"]:
            shutil.copy(DATA_DIR / f"{split}_{field}.parquet", sandbox / "data")
        shutil.copy(DATA_DIR / "splits.json", sandbox / "data")
        shutil.copy(ENGINE_FILE, sandbox / "engine.py")
        shutil.copy(RUNNER_FILE, sandbox / "runner.py")
        shutil.copy(strategy_path, sandbox / "strategy.py")

        result = subprocess.run(
            [sys.executable, "runner.py", "strategy.py", json.dumps(params), "data", split, str(cost_bps)],
            capture_output=True, text=True, cwd=sandbox, timeout=timeout,
            env=_scrubbed_env(sandbox),
        )
        if not result.stdout.strip():
            return {"ok": False, "error": f"returncode={result.returncode} stderr={(result.stderr or '(empty)')[-500:]}"}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

