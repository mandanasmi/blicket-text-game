"""Streamlit Cloud entry point: ACTIVE mixed-rule experiment (mixed_and_or).

Extension experiment link `nexiom-text-game-active-conjunctive`.
Reuses active_app/app.py, pinning the main-game rule to mixed_and_or. Firebase comes
from this deployment's own secrets ([firebase] block), so it writes to its own DB.
"""
import os
import sys

os.environ["NEXIOM_MAIN_RULE"] = "mixed_and_or"
os.environ["NEXIOM_CONDITION"] = "active_mixed_and_or"
os.environ["NEXIOM_EXTENSION_QUESTIONS"] = "1"
# Test history stays visible during Q&A; instead the study is proctored: full-screen
# gate, focus/tab-switch logging, copy/paste blocking, plus the bonus instructions page.
os.environ["NEXIOM_PROCTORED"] = "1"
# Set explicitly (not just removed): Cloud redeploys keep the old process env.
os.environ["NEXIOM_HIDE_HISTORY_IN_QA"] = "0"
os.environ["NEXIOM_VARIED_NEXIOMS"] = "1"
# Tell participants some combination always turns the machine on (keeps them searching).
os.environ["NEXIOM_GUARANTEE_HINT"] = "1"
# Mixed-rule study: 8 objects, 3 Nexioms, 32 tests.
os.environ["NEXIOM_VARIED_OBJECT_COUNTS"] = "8"
os.environ["NEXIOM_VARIED_NEXIOM_COUNTS"] = "3"
# Prolific completion code for this study.
os.environ["SURVEY_COMPLETION_CODE"] = "C1IK38P3"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ACTIVE = os.path.join(_ROOT, "active_app")

for p in (_ACTIVE, _ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# Streamlit doesn't reload this helper when a redeploy changes it, so evict a stale copy.
_fm = sys.modules.get("_fresh_modules")
if _fm is not None and os.path.getmtime(_fm.__file__) > getattr(_fm, "LOADED_MTIME", 0):
    del sys.modules["_fresh_modules"]
from _fresh_modules import drop_changed_modules, run_app, stamp_modules

# Reload active_app/ and env/ modules that changed in a redeploy (Streamlit won't).
drop_changed_modules(_ROOT)
try:
    run_app(os.path.join(_ACTIVE, "app.py"))
finally:
    stamp_modules(_ROOT)
