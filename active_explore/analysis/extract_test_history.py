"""
Extract test history from participant JSONs in the data folder.
Output format: test1 -> object 2 -> machine on
              test2 -> object 1, object 3 -> machine off
              ...

Reads state_history from each participant JSON. Object indices are 0-based
in the JSON; we display as 1-based (object 1, object 2, ...) to match the game UI.
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default: data folder is sibling of analysis/ (i.e. active_explore/data/)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "test_histories")


def format_objects(indices):
    """Convert 0-based object indices to 'object 1', 'object 2', ... (1-based)."""
    if not indices:
        return "none"
    return ", ".join(f"object {i + 1}" for i in sorted(indices))


def format_machine(lit):
    """Convert machine_lit bool to 'machine on' / 'machine off'."""
    return "machine on" if lit else "machine off"


def test_history_lines(state_history):
    """Yield one line per test: testN -> object X, ... -> machine on/off."""
    if not state_history:
        return
    for i, state in enumerate(state_history):
        if not isinstance(state, dict):
            continue
        step = state.get("step_number", i + 1)
        objs = state.get("objects_on_machine")
        if not isinstance(objs, list):
            objs = []
        lit = state.get("machine_lit", False)
        obj_str = format_objects(objs)
        machine_str = format_machine(lit)
        yield f"test{step} -> {obj_str} -> {machine_str}"


def main():
    parser = argparse.ArgumentParser(description="Extract test history from participant JSONs.")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Folder containing participant JSON files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Folder to write test history text files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing files.")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skip {name}: {e}")
            continue
        state_history = data.get("state_history")
        if not state_history:
            continue
        pid = data.get("participant_id", name.replace(".json", ""))
        lines = list(test_history_lines(state_history))
        if not lines:
            continue
        if args.stdout:
            print(f"=== {pid} ===")
            for line in lines:
                print(line)
            print()
        else:
            out_path = os.path.join(args.output_dir, f"{pid}_test_history.txt")
            with open(out_path, "w") as f:
                f.write("\n".join(lines))
                f.write("\n")
            count += 1
    if not args.stdout:
        print(f"Wrote {count} test history files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
