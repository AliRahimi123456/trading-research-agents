import tempfile
from pathlib import Path

from src.config import PRIVATE
from src.isolation import run_isolated

# What a strategy actually receives: no credentials, no path back to the real project.
REALISTIC_PROBE = """
import os, glob

def target_weights(data, **kwargs):
    suspicious_env = sorted(k for k in os.environ if any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET")))
    local_files = sorted(glob.glob("*")) + sorted(glob.glob("data/*"))
    raise RuntimeError(f"ENV_LEAKS={suspicious_env} LOCAL_FILES={local_files}")
"""

# What happens if a strategy is (unrealistically) HANDED the real absolute path anyway --
# proving this is process isolation, not an OS sandbox that blocks arbitrary file access.
ABS_PATH_PROBE = f"""
import glob

def target_weights(data, **kwargs):
    files = glob.glob(r"{PRIVATE}\\\\**\\\\*", recursive=True)
    raise RuntimeError(f"FILES_FOUND_VIA_HARDCODED_ABS_PATH={{len(files)}}: {{files[:3]}}")
"""


def _write_temp_strategy(code: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".py")
    Path(path).write_text(code)
    return Path(path)


if __name__ == "__main__":
    realistic = run_isolated(_write_temp_strategy(REALISTIC_PROBE), {}, "dev")
    print("realistic probe (env vars + the sandbox's own relative-path view):")
    print(" ", realistic)
    assert "ENV_LEAKS=[]" in realistic["error"], "credential-like env vars reachable from sandbox!"
    print("-> PASSED: no credentials leaked, only the 4 staged files are visible\n")

    absolute = run_isolated(_write_temp_strategy(ABS_PATH_PROBE), {}, "dev")
    print("absolute-path probe (deliberately handed the real private/ path):")
    print(" ", absolute)
    print("-> expected to find files: this is process isolation, not an OS sandbox.")
    print("   nothing in the strategy's real inputs (argv, env, or its own sandbox")
    print("   folder) ever reveals this path -- it only works because we, the")
    print("   trusted caller, hardcoded it into the probe ourselves.")
