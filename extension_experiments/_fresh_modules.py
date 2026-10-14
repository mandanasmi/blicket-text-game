"""Keep the repo's own modules fresh across Streamlit Cloud redeploys.

Streamlit only reloads modules that live next to the main script (this folder). The
extension wrappers run active_app/app.py, so after a git pull the running process keeps
the old active_app/ and env/ modules in sys.modules, and app.py can fail to import names
that only exist in the new code. drop_changed_modules() evicts any repo module whose file
changed since stamp_modules() last ran, so the next import loads the new code.

run_app() runs the shared app script cheaply on every rerun (see its docstring).
"""
import builtins
import os
import sys

_STAMP = "_nexiom_loaded_mtime"
# The wrappers compare this to the file's mtime to reload this helper after a redeploy.
LOADED_MTIME = os.path.getmtime(__file__)

# Repo modules found in sys.modules, rebuilt only when sys.modules changes size, so
# each rerun doesn't walk (and abspath) every loaded module.
_repo_cache = {"key": None, "mods": []}
# Compiled app script, reused until the file's mtime changes.
_code_cache = {}


def _repo_modules(root):
    root = os.path.abspath(root) + os.sep
    key = (root, len(sys.modules))
    if _repo_cache["key"] != key:
        mods = []
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if not path:
                continue
            path = os.path.abspath(path)
            if path.startswith(root) and "site-packages" not in path:
                mods.append((name, module, path))
        _repo_cache["key"], _repo_cache["mods"] = key, mods
    return list(_repo_cache["mods"])


def drop_changed_modules(root):
    for name, module, path in _repo_modules(root):
        stamped = getattr(module, _STAMP, None)
        try:
            changed = stamped is not None and os.path.getmtime(path) > stamped
        except OSError:
            changed = True
        if changed:
            sys.modules.pop(name, None)
            _repo_cache["key"] = None  # re-imported module must be re-stamped


def stamp_modules(root):
    for _, module, path in _repo_modules(root):
        if getattr(module, _STAMP, None) is None:
            try:
                setattr(module, _STAMP, os.path.getmtime(path))
            except (OSError, AttributeError):
                pass


def run_app(path):
    """Run the app script as __main__, like runpy.run_path, with less work per rerun.

    - Compiles the script once and reuses the code object until the file changes
      (runpy re-reads and re-compiles the ~1500-line app.py on every click).
    - Turns off Streamlit's forced full gc.collect() after every rerun
      (runner.postScriptGC), which was about half of the server CPU per click and got
      these deployments CPU-throttled on Streamlit Cloud. Python's normal automatic
      garbage collection still runs.
    """
    try:
        from streamlit import config as st_config
        if st_config.get_option("runner.postScriptGC"):
            st_config.set_option("runner.postScriptGC", False)
    except Exception:
        pass

    mtime = os.path.getmtime(path)
    cached = _code_cache.get(path)
    if cached is None or cached[0] != mtime:
        with open(path, "rb") as f:
            cached = (mtime, compile(f.read(), path, "exec"))
        _code_cache[path] = cached
    exec(cached[1], {"__name__": "__main__", "__file__": path, "__builtins__": builtins})
