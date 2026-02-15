"""
Extract all participants' active exploration action history and test history
from human_active_data_no_prior_experience.json (main_game only).
Writes one JSON per participant into active_explore/ with:
  - action_history: list of action strings
  - state_history: list of test states (objects_on_machine, machine_lit, step_number)
  - test_timings: list of {test_number, time_since_previous_seconds, ...}
  - user_test_actions: list of test actions (objects_tested, machine_state_after, ...)
  - chosen_blickets: participant's chosen blicket indices (user_chosen_blickets)
  - chosen_rule: participant's chosen rule type (e.g. "Conjunctive (ALL Nexioms must be present)")
  - rule_hypothesis: participant's free-text rule explanation (if present)
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "human_active_data_no_prior_experience.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "active_explore")


def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    count = 0
    for pid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        mg = entry.get("main_game")
        if not isinstance(mg, dict):
            continue
        action_history = mg.get("action_history")
        state_history = mg.get("state_history")
        test_timings = mg.get("test_timings")
        user_test_actions = mg.get("user_test_actions")
        if action_history is None and state_history is None:
            continue
        out = {
            "participant_id": pid,
            "action_history": action_history if isinstance(action_history, list) else [],
            "state_history": state_history if isinstance(state_history, list) else [],
            "test_timings": test_timings if isinstance(test_timings, list) else [],
            "user_test_actions": user_test_actions if isinstance(user_test_actions, list) else [],
            "chosen_blickets": mg.get("user_chosen_blickets") if isinstance(mg.get("user_chosen_blickets"), list) else [],
            "chosen_rule": mg.get("rule_type") or mg.get("rule") or None,
        }
        if mg.get("rule_hypothesis"):
            out["rule_hypothesis"] = mg["rule_hypothesis"]
        # Ground truth for context
        if mg.get("true_rule"):
            out["true_rule"] = mg["true_rule"]
        if mg.get("config"):
            out["true_blicket_indices"] = mg["config"].get("blicket_indices")
        out_path = os.path.join(OUTPUT_DIR, f"{pid}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        count += 1
    print(f"Extracted {count} participants to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
