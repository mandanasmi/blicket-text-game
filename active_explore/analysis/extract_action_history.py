"""
Extract action history from participant JSONs in the data folder.
Output format:
    action 1: Placed Object 2 on machine,
    action 2: Test the machine -> Nexiom machine is ON,
    action 3: Removed Object 2 from machine,
    ...

Converts "Test Result: Nexiom machine is ON/OFF" to
"Test the machine -> Nexiom machine is ON/OFF". Other actions (Placed/Removed)
are written as-is.
"""

import argparse
import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "action_histories")

# Match "Test Result: Nexiom machine is ON" or "... is OFF"
TEST_RESULT_PATTERN = re.compile(r"^Test Result:\s*(.+)$", re.IGNORECASE)


def format_action(raw):
    """Convert raw action string; 'Test Result: ...' becomes 'Test the machine -> ...'."""
    raw = (raw or "").strip()
    m = TEST_RESULT_PATTERN.match(raw)
    if m:
        outcome = m.group(1).strip()
        return f"Test the machine -> {outcome}"
    return raw


def action_history_lines(action_history):
    """Yield one line per action: action N: ..., (with trailing comma)."""
    if not action_history:
        return
    for i, raw in enumerate(action_history):
        if not isinstance(raw, str):
            raw = str(raw)
        line = format_action(raw)
        yield f"action {i + 1}: {line},"


def main():
    parser = argparse.ArgumentParser(description="Extract action history from participant JSONs.")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Folder containing participant JSON files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Folder to write action history text files (default: {DEFAULT_OUT_DIR})",
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
        action_history = data.get("action_history")
        if not action_history:
            continue
        pid = data.get("participant_id", name.replace(".json", ""))
        lines = list(action_history_lines(action_history))
        if not lines:
            continue
        if args.stdout:
            print(f"=== {pid} ===")
            for line in lines:
                print(f"    {line}")
            print()
        else:
            out_path = os.path.join(args.output_dir, f"{pid}_action_history.txt")
            with open(out_path, "w") as f:
                for line in lines:
                    f.write(f"    {line}\n")
            count += 1
    if not args.stdout:
        print(f"Wrote {count} action history files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
