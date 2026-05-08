"""Root entry point for Streamlit Cloud.

The real app lives in active_app/app.py. This shim makes the repo root work as
the configured "Main file path" on Streamlit Cloud without needing a dashboard
change.
"""
import os
import sys
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_ACTIVE = os.path.join(_HERE, "active_app")

for p in (_ACTIVE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

runpy.run_path(os.path.join(_ACTIVE, "app.py"), run_name="__main__")
