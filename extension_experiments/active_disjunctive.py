"""Streamlit Cloud entry point: ACTIVE experiment, DISJUNCTIVE rule only.

Extension experiment link `nexiom-text-game-active-disjunctive`.
Reuses active_app/app.py, pinning the main-game rule to disjunctive. Firebase comes
from this deployment's own secrets ([firebase] block), so it writes to its own DB.
"""
import os
import sys
import runpy

os.environ["NEXIOM_MAIN_RULE"] = "disjunctive"
os.environ["NEXIOM_CONDITION"] = "active_disjunctive"
os.environ["NEXIOM_EXTENSION_QUESTIONS"] = "1"
os.environ["NEXIOM_HIDE_HISTORY_IN_QA"] = "1"
os.environ["NEXIOM_VARIED_NEXIOMS"] = "1"
# Tell participants some combination always turns the machine on (keeps them searching).
os.environ["NEXIOM_GUARANTEE_HINT"] = "1"
# For now only 8 objects with all 8 as Nexioms. Use NEXIOM_VARIED_NEXIOM_COUNTS = "2,4,8"
# for 2/8, 4/8, 8/8, and NEXIOM_VARIED_OBJECT_COUNTS = "4,8" to add 1/4, 3/4, 4/4 back.
os.environ["NEXIOM_VARIED_OBJECT_COUNTS"] = "8"
os.environ["NEXIOM_VARIED_NEXIOM_COUNTS"] = "8"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ACTIVE = os.path.join(_ROOT, "active_app")

for p in (_ACTIVE, _ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _fresh_modules import drop_changed_modules, stamp_modules

# Reload active_app/ and env/ modules that changed in a redeploy (Streamlit won't).
drop_changed_modules(_ROOT)
try:
    runpy.run_path(os.path.join(_ACTIVE, "app.py"), run_name="__main__")
finally:
    stamp_modules(_ROOT)
