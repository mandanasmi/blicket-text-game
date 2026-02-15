"""
Single entry point to run round7 data extraction and plot generation.

Usage:
  python prepare_results_human_data.py <task> [task_options...]
  python prepare_results_human_data.py --list

Tasks:
  extract_main_game       Build main_game_data_with_prior_experience.csv from data.json
  extract_active_explore  Build per-participant JSONs in active_explore/ from human_active_data_no_prior_experience.json
  plot_accuracy_by_rule   Produce accuracy_by_rule_type_grouped.png from main_game CSV
  plot_correlation_scatters Produce accuracy_correlation_scatters.png from data.json
  plot_human_llm          Produce accuracy_human_llm_by_rule_type.png (human + LLM)

Any arguments after the task name are passed to the underlying script (e.g. --input, --output).
Examples:
  python prepare_results_human_data.py extract_main_game --input data.json --output main_game_data_with_prior_experience.csv
  python prepare_results_human_data.py plot_accuracy_by_rule --input main_game_data_with_prior_experience.csv --output accuracy_by_rule_type_grouped.png
  python prepare_results_human_data.py plot_human_llm --output accuracy_human_llm_by_rule_type.png
"""

import importlib.util
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = {
    "extract_main_game": {
        "module": "extract_main_game_data_with_prior_experience",
        "description": "Build main_game_data_with_prior_experience.csv from data.json",
    },
    "extract_active_explore": {
        "module": "extract_active_exploration_sequences",
        "description": "Build per-participant JSONs in active_explore/ from human_active_data_no_prior_experience.json",
    },
    "plot_accuracy_by_rule": {
        "module": "plot_accuracy_by_rule_type_grouped",
        "description": "Produce accuracy_by_rule_type_grouped.png from main_game CSV",
        "subdir": "accuracy",
    },
    "plot_correlation_scatters": {
        "module": "plot_accuracy_correlation_scatters",
        "description": "Produce accuracy_correlation_scatters.png from data.json",
        "subdir": "accuracy",
    },
    "plot_human_llm": {
        "module": "plot_accuracy_human_llm_by_rule_type",
        "description": "Produce accuracy_human_llm_by_rule_type.png (human + LLM agents)",
    },
}


def _load_and_run(module_name, argv, subdir=""):
    """Load module from round7 (or subdir) and run its main (or execute module-level code)."""
    base = os.path.join(SCRIPT_DIR, subdir) if subdir else SCRIPT_DIR
    module_path = os.path.join(base, f"{module_name}.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    # So the module sees our argv when it parses args
    old_argv = sys.argv
    sys.argv = [module_path] + argv
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
    finally:
        sys.argv = old_argv


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("Tasks:")
        for name, info in TASKS.items():
            print(f"  {name:28} {info['description']}")
        return 0
    if args[0] == "--list":
        for name, info in TASKS.items():
            print(f"  {name:28} {info['description']}")
        return 0
    task = args[0]
    task_args = args[1:]
    if task not in TASKS:
        print(f"Unknown task: {task}", file=sys.stderr)
        print("Use --list to see available tasks.", file=sys.stderr)
        return 1
    task_info = TASKS[task]
    module_name = task_info["module"]
    subdir = task_info.get("subdir", "")
    try:
        _load_and_run(module_name, task_args, subdir=subdir)
        return 0
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running {task}: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main() or 0)
