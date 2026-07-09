"""Streamlit Cloud entry point: PASSIVE experiment, DISJUNCTIVE histories only.

Extension experiment link `nexiom-text-game-passive-disjunctive`.
Reuses passive_app/app.py. Respondents are rotated round-robin across the recorded
action histories in the disjunctive histories folder. Firebase comes from this
deployment's own secrets ([firebase] block), so it writes to its own DB.

Point this at your disjunctive histories folder in ONE of these ways (first wins):
  1. Streamlit Cloud env var  NEXIOM_PASSIVE_HISTORY_DIR=/path/to/disjunctive
  2. Streamlit secret         [passive]\n  history_dir = "/path/to/disjunctive"
  3. Drop *.txt histories into extension_experiments/histories/disjunctive/ (default below)
Until a non-empty folder is found, it falls back to the single OG conjunctive history.
"""
import os
import sys
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PASSIVE = os.path.join(_ROOT, "passive_app")

os.environ["NEXIOM_CONDITION"] = "passive_disjunctive"
# Default rotation folder; override via NEXIOM_PASSIVE_HISTORY_DIR or [passive].history_dir.
os.environ.setdefault(
    "NEXIOM_PASSIVE_HISTORY_DIR",
    os.path.join(_HERE, "histories", "disjunctive"),
)

for p in (_PASSIVE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

runpy.run_path(os.path.join(_PASSIVE, "app.py"), run_name="__main__")
