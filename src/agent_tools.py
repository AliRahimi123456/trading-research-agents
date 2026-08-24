"""Thin @tool wrappers around the plain functions in registry.py.

Kept separate from registry.py on purpose: our own check_v1.py/check_v2.py/check_v3.py
scripts call sweep()/read_registry()/record_decision() as plain Python functions.
LangChain's @tool decorator turns a function into a BaseTool object, which changes how
it's called (agents call .invoke(), not the function directly) -- wrapping here instead
of decorating in place means neither calling convention breaks the other.
"""
from langchain.tools import tool

from src.registry import sweep as _sweep, read_registry as _read_registry, record_decision as _record_decision


@tool
def sweep(version: str, grid_json: str, note: str = "") -> str:
    """Backtest strategies/<version>.py over several parameter sets in ONE call.

    version   : file stem, e.g. "v1" for strategies/v1.py
    grid_json : JSON list of parameter objects, e.g. [{"top_n":3},{"top_n":4}]
    note      : short reason for this sweep

    Runs each configuration in an isolated subprocess. Returns a CSV table sorted by
    validation Sharpe. Max 12 configurations per version, cumulative. Every row is
    written to registry.csv, including failures. vN is blocked until v(N-1) is swept,
    reviewed and decided.
    """
    return _sweep(version, grid_json, note)


@tool
def read_registry(version: str = "") -> str:
    """Every run recorded so far as CSV, accepted and rejected. Pass a version to filter."""
    return _read_registry(version)


@tool
def record_decision(version: str, champion: str, rationale: str, params_json: str) -> str:
    """Record the approved outcome of a version. REQUIRED before the next version can be swept.

    version    : the version just reviewed, e.g. "v2"
    champion   : which version is champion after applying the selection rule
    rationale  : cite the selection rule and the specific numbers that decided it
    params_json: the champion's parameters as JSON
    """
    return _record_decision(version, champion, rationale, params_json)
