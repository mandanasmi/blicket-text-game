"""Streamlit Cloud entry point: ACTIVE experiment, CONJUNCTIVE rule only.

Extension experiment link `nexiom-text-game-active-conjunctive`.
Reuses active_app/app.py, pinning the main-game rule to conjunctive. Firebase comes
from this deployment's own secrets ([firebase] block), so it writes to its own DB.
"""
import os
import sys
import runpy

os.environ["NEXIOM_MAIN_RULE"] = "conjunctive"
os.environ["NEXIOM_CONDITION"] = "active_conjunctive"
os.environ["NEXIOM_EXTENSION_QUESTIONS"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ACTIVE = os.path.join(_ROOT, "active_app")

for p in (_ACTIVE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

runpy.run_path(os.path.join(_ACTIVE, "app.py"), run_name="__main__")
