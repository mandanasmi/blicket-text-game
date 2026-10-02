"""Keep the repo's own modules fresh across Streamlit Cloud redeploys.

Streamlit only reloads modules that live next to the main script (this folder). The
extension wrappers run active_app/app.py, so after a git pull the running process keeps
the old active_app/ and env/ modules in sys.modules, and app.py can fail to import names
that only exist in the new code. drop_changed_modules() evicts any repo module whose file
changed since stamp_modules() last ran, so the next import loads the new code.
"""
import os
import sys

_STAMP = "_nexiom_loaded_mtime"


def _repo_modules(root):
    root = os.path.abspath(root) + os.sep
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        path = os.path.abspath(path)
        if path.startswith(root) and "site-packages" not in path:
            yield name, module, path


def drop_changed_modules(root):
    for name, module, path in _repo_modules(root):
        stamped = getattr(module, _STAMP, None)
        try:
            changed = stamped is not None and os.path.getmtime(path) > stamped
        except OSError:
            changed = True
        if changed:
            sys.modules.pop(name, None)


def stamp_modules(root):
    for _, module, path in _repo_modules(root):
        if getattr(module, _STAMP, None) is None:
            try:
                setattr(module, _STAMP, os.path.getmtime(path))
            except (OSError, AttributeError):
                pass
