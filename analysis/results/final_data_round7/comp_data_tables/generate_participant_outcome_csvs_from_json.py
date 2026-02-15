"""
Generate participant outcome CSVs directly from human_active_data_no_prior_experience.json.
Outputs:
- participants_both_correct.csv
- participants_both_wrong.csv
- participants_rule_correct_objects_wrong.csv
- participants_rule_wrong_objects_correct.csv
"""

import argparse
import json
from datetime import datetime

import pandas as pd


def get_prolific_id(entry):
    """Extract prolific_id from an entry."""
    if "demographics" in entry and isinstance(entry["demographics"], dict):
        if "prolific_id" in entry["demographics"]:
            return entry["demographics"]["prolific_id"]
    if "config" in entry and isinstance(entry["config"], dict):
        cfg = entry["config"]
        if "demographics" in cfg and isinstance(cfg["demographics"], dict):
            if "prolific_id" in cfg["demographics"]:
                return cfg["demographics"]["prolific_id"]
    return None


def parse_rule_type(rule_type_str):
    """Parse rule type from string to 'conjunctive' or 'disjunctive'."""
    if not rule_type_str or not isinstance(rule_type_str, str):
        return None
    s = rule_type_str.lower()
    if "conjunctive" in s or "all" in s:
        return "conjunctive"
    if "disjunctive" in s or "any" in s:
        return "disjunctive"
    return None


def get_user_chosen_blickets(round_data):
    """Extract user's chosen blickets from round data."""
    if "user_chosen_blickets" in round_data:
        chosen = round_data["user_chosen_blickets"]
        if isinstance(chosen, list):
            chosen = [x for x in chosen if x is not None]
            return sorted(chosen)
    if "blicket_classifications" in round_data:
        classifications = round_data["blicket_classifications"]
        if isinstance(classifications, dict):
            chosen = []
            for key, value in classifications.items():
                if value == "Yes":
                    try:
                        idx = int(key.split("_")[1])
                        chosen.append(idx)
                    except (ValueError, IndexError):
                        pass
            return sorted(chosen)
    return None


def _get_true_blickets(round_data):
    """Ground truth blicket indices."""
    out = round_data.get("true_blicket_indices")
    if out is not None:
        return out
    cfg = round_data.get("config") or {}
    return cfg.get("blicket_indices")


def _get_true_rule(round_data):
    """Ground truth rule."""
    out = round_data.get("true_rule")
    if out:
        return out
    cfg = round_data.get("config") or {}
    out = cfg.get("rule")
    if out:
        return out
    return round_data.get("rule") or ""


def _parse_duration_from_start_end(round_data):
    """Compute total_time_seconds from start_time/end_time."""
    start_s = round_data.get("start_time")
    end_s = round_data.get("end_time")
    if not start_s or not end_s:
        return None
    try:
        start = datetime.fromisoformat(start_s.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_s.replace("Z", "+00:00"))
        return round((end - start).total_seconds(), 3)
    except (ValueError, TypeError):
        return None


def _get_total_round_time(round_data):
    """Total round time."""
    t = round_data.get("total_time_seconds")
    if t is not None:
        return round(float(t), 3)
    return _parse_duration_from_start_end(round_data)


def extract_rows(data):
    """Extract one row per round from JSON data."""
    results = []
    for user_id, user_data in data.items():
        if not isinstance(user_data, dict):
            continue
        prolific_id = get_prolific_id(user_data) or user_id
        main_game = user_data.get("main_game", {})
        if not main_game or not isinstance(main_game, dict):
            continue

        has_nested_rounds = any(
            k.startswith("round_") for k in main_game.keys() if isinstance(k, str)
        )

        if has_nested_rounds:
            for round_key, round_data in main_game.items():
                if not isinstance(round_data, dict) or not round_key.startswith("round_"):
                    continue
                round_number = round_data.get("round_number", 0)
                if round_number == 0:
                    continue
                row = _extract_round_row(prolific_id, False, round_data, round_number)
                if row:
                    results.append(row)
        else:
            row = _extract_round_row(prolific_id, False, main_game, 1)
            if row:
                results.append(row)

    return results


def _extract_round_row(prolific_id, has_prior, round_data, round_number):
    """Extract a single round row."""
    test_timings = round_data.get("test_timings", [])
    num_tests = len(test_timings)

    time_per_test = []
    if isinstance(test_timings, list):
        for t in test_timings:
            if isinstance(t, dict) and "time_since_previous_seconds" in t:
                time_per_test.append(round(t["time_since_previous_seconds"], 3))

    total_test_time = sum(time_per_test) if time_per_test else None
    total_round_time = _get_total_round_time(round_data)
    exploration_time_seconds = None
    if isinstance(test_timings, list) and len(test_timings) > 0:
        last_timing = test_timings[-1]
        if isinstance(last_timing, dict):
            exploration_time_seconds = last_timing.get("time_since_start_seconds")
            if exploration_time_seconds is not None:
                exploration_time_seconds = round(float(exploration_time_seconds), 3)
    true_blickets = _get_true_blickets(round_data)
    true_rule = _get_true_rule(round_data)
    user_chosen_blickets = get_user_chosen_blickets(round_data)
    rule_hypothesis = round_data.get("rule_hypothesis", "")
    rule_type_str = round_data.get("rule_type", "")
    user_rule_choice = parse_rule_type(rule_type_str)

    object_accuracy = None
    if true_blickets is not None and user_chosen_blickets is not None:
        true_set = set(true_blickets) if isinstance(true_blickets, list) else set()
        user_set = set(user_chosen_blickets) if isinstance(user_chosen_blickets, list) else set()
        object_accuracy = true_set == user_set

    rule_accuracy = None
    if true_rule and user_rule_choice:
        rule_accuracy = true_rule.lower() == user_rule_choice.lower()

    action_history_length = round_data.get("action_history_length")
    if action_history_length is None:
        action_history = round_data.get("action_history", [])
        action_history_length = len(action_history) if isinstance(action_history, list) else None

    return {
        "user_id": prolific_id,
        "has_prior_experience": has_prior,
        "round_number": round_number,
        "num_tests": num_tests,
        "time_per_test_seconds": str(time_per_test),
        "total_test_time_seconds": round(total_test_time, 3) if total_test_time else None,
        "total_round_time_seconds": round(total_round_time, 3) if total_round_time else None,
        "exploration_time_seconds": exploration_time_seconds,
        "user_object_identification": (
            str(sorted(user_chosen_blickets)) if user_chosen_blickets else "[]"
        ),
        "ground_truth_objects": str(sorted(true_blickets)) if true_blickets else "[]",
        "object_identification_correct": object_accuracy,
        "rule_inference_text": rule_hypothesis,
        "rule_choice": user_rule_choice if user_rule_choice else "",
        "ground_truth_rule": true_rule,
        "rule_choice_correct": rule_accuracy,
        "action_history_length": action_history_length,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate participant outcome CSVs from JSON."
    )
    parser.add_argument(
        "--input",
        default="../human_active_data_no_prior_experience.json",
        help="Input JSON file (default: from round7)",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory (default: comp_data_tables)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    rows = extract_rows(data)
    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("ERROR: No data extracted.")
        return

    df = df.sort_values(["user_id", "round_number"]).reset_index(drop=True)

    # Classify outcomes
    obj_ok = df["object_identification_correct"] == True
    rule_ok = df["rule_choice_correct"] == True
    both_ok = obj_ok & rule_ok
    both_wrong = ~obj_ok & ~rule_ok
    rule_ok_obj_wrong = rule_ok & ~obj_ok
    rule_wrong_obj_ok = ~rule_ok & obj_ok

    outputs = [
        ("participants_both_correct.csv", both_ok),
        ("participants_both_wrong.csv", both_wrong),
        ("participants_rule_correct_objects_wrong.csv", rule_ok_obj_wrong),
        ("participants_rule_wrong_objects_correct.csv", rule_wrong_obj_ok),
    ]

    out_dir = args.out_dir.rstrip("/")
    drop_cols = ["round_number", "has_prior_experience"]
    col_order = ["user_id", "num_tests", "action_history_length", "time_per_test_seconds",
                 "total_test_time_seconds", "exploration_time_seconds", "total_round_time_seconds",
                 "user_object_identification",
                 "ground_truth_objects", "object_identification_correct", "rule_inference_text",
                 "rule_choice", "ground_truth_rule", "rule_choice_correct"]
    for fname, mask in outputs:
        sub = df[mask].drop(columns=drop_cols, errors="ignore")
        sub = sub[[c for c in col_order if c in sub.columns]]
        path = f"{out_dir}/{fname}"
        sub.to_csv(path, index=False)
        print(f"Saved: {path} ({len(sub)} rows)")

    exploration_cols = ["user_id", "exploration_time_seconds", "total_test_time_seconds", "total_round_time_seconds"
                       ] + [c for c in df.columns if c not in ("user_id", "exploration_time_seconds", "total_test_time_seconds", "total_round_time_seconds")]
    exploration_csv = df[[c for c in exploration_cols if c in df.columns]].drop(columns=drop_cols, errors="ignore").copy()
    exploration_path = f"{out_dir}/exploration_time_seconds.csv"
    exploration_csv.to_csv(exploration_path, index=False)
    print(f"Saved: {exploration_path} ({len(exploration_csv)} rows)")


if __name__ == "__main__":
    main()
