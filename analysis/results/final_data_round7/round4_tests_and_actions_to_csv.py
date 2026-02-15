"""
Extract data.json to CSVs with test sequences and individual actions.
Supports round4 and round7 (flat main_game with action_history, state_history, test_timings).

Output: {out_prefix}_test_sequence.csv, {out_prefix}_actions.csv
"""

import argparse
import json
import os

import pandas as pd


def _get_user_id(entry, pid):
    d = entry.get("demographics") or {}
    if isinstance(d, dict) and d.get("prolific_id"):
        return d["prolific_id"]
    return pid


def main():
    ap = argparse.ArgumentParser(description="Extract tests + actions to CSV")
    ap.add_argument("--input", default=None, help="Path to data.json")
    ap.add_argument("--round", type=int, default=None, help="Use round N: auto-set input and output prefix (e.g. 4 or 7)")
    ap.add_argument("--out-prefix", default=None, help="Output file prefix (default: round4 or from --round)")
    ap.add_argument("--out-dir", default=".", help="Output directory")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.round is not None:
        r = args.round
        inp = os.path.normpath(os.path.join(script_dir, "..", f"round{r}", "data.json"))
        prefix = args.out_prefix if args.out_prefix else f"round{r}"
    elif args.input:
        inp = os.path.abspath(args.input)
        prefix = args.out_prefix if args.out_prefix else "round4"
    else:
        inp = os.path.normpath(os.path.join(script_dir, "..", "round4", "data.json"))
        prefix = args.out_prefix if args.out_prefix else "round4"

    with open(inp, "r") as f:
        data = json.load(f)

    rows_tests = []
    rows_actions = []

    for pid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        mg = entry.get("main_game") or {}
        if not isinstance(mg, dict):
            continue
        ah = mg.get("action_history") or []
        sh = mg.get("state_history") or []
        tt = mg.get("test_timings") or []
        if not isinstance(ah, list) or not isinstance(sh, list):
            continue

        uid = _get_user_id(entry, pid)

        # Test sequence: one row per test
        for i, s in enumerate(sh):
            if not isinstance(s, dict):
                continue
            objs = s.get("objects_on_machine")
            if isinstance(objs, list):
                objs_str = json.dumps(objs)
            else:
                objs_str = json.dumps([])
            lit = s.get("machine_lit", None)
            step = s.get("step_number", i + 1)
            t_prev = None
            t_start = None
            if isinstance(tt, list) and i < len(tt):
                t = tt[i]
                if isinstance(t, dict):
                    t_prev = t.get("time_since_previous_seconds")
                    t_start = t.get("time_since_start_seconds")
            rows_tests.append({
                "user_id": uid,
                "test_index": i + 1,
                "step_number": step,
                "objects_on_machine": objs_str,
                "machine_lit": lit,
                "time_since_previous_seconds": t_prev,
                "time_since_start_seconds": t_start,
            })

        # Actions: one row per action; assign test_index by "Test Result" boundaries
        test_idx = 1
        for j, text in enumerate(ah):
            if isinstance(text, str) and text.strip().lower().startswith("test result"):
                rows_actions.append({
                    "user_id": uid,
                    "action_index": j + 1,
                    "action_text": text,
                    "test_index": test_idx,
                })
                test_idx += 1
            else:
                rows_actions.append({
                    "user_id": uid,
                    "action_index": j + 1,
                    "action_text": text if isinstance(text, str) else str(text),
                    "test_index": test_idx,
                })

    out_dir = args.out_dir.rstrip("/")
    path_tests = os.path.join(out_dir, f"{prefix}_test_sequence.csv")
    path_actions = os.path.join(out_dir, f"{prefix}_actions.csv")

    df_tests = pd.DataFrame(rows_tests)
    df_actions = pd.DataFrame(rows_actions)

    df_tests.to_csv(path_tests, index=False)
    df_actions.to_csv(path_actions, index=False)

    print(f"Saved: {path_tests} ({len(df_tests)} rows)")
    print(f"Saved: {path_actions} ({len(df_actions)} rows)")


if __name__ == "__main__":
    main()
