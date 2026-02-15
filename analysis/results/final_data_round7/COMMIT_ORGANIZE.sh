#!/usr/bin/env bash
# Organize round7 (final_data_round7) into logical commits.
# Run from repo root: bash analysis/results/final_data_round7/COMMIT_ORGANIZE.sh

set -e
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASE="analysis/results/final_data_round7"

commit_if_staged() {
  local msg="$1"
  if git diff --cached --quiet; then
    echo "(skip: nothing staged for: $msg)"
  else
    git commit -m "$msg"
  fi
}

# 1. Accuracy scripts and outputs
git add "$BASE/accuracy/" 2>/dev/null || true
commit_if_staged "Round7: accuracy scripts and plots (grouped, accuracy-only, correlation scatters, four outcomes)"

# 2. Exploration time
git add "$BASE/exploration_time/" 2>/dev/null || true
commit_if_staged "Round7: exploration time scripts and outputs (actions/tests/time by rule)"

# 3. Human search strategy
git add "$BASE/human_search_strategy/" 2>/dev/null || true
commit_if_staged "Round7: human search strategy (action sequences, efficient strategies, maxed-out by rule)"

# 4. Info-gain and hypothesis
git add "$BASE/infogain_hypothesis/" 2>/dev/null || true
commit_if_staged "Round7: infogain and hypothesis scripts and outputs (run_infogain_hypothesis entrypoint)"

# 5. PNAS plots
git add "$BASE/pnas/" 2>/dev/null || true
commit_if_staged "Round7: PNAS comparison plots (run_pnas_plots entrypoint)"

# 6. Correlations
git add "$BASE/correlations/" 2>/dev/null || true
commit_if_staged "Round7: correlation data and plots (no-prior-102, best performers)"

# 7. Active exploration (comp_data_tables and analysis)
git add "$BASE/active_explore/" 2>/dev/null || true
commit_if_staged "Round7: active exploration (comp_data_tables, participant outcomes, ranked successful)"

# 8. Root-level scripts and shared files in final_data_round7 (including this script)
git add "$BASE/"*.py "$BASE/"*.tex "$BASE/COMMIT_ORGANIZE.sh" 2>/dev/null || true
commit_if_staged "Round7: root scripts and shared assets (prepare_results, extraction, LaTeX definitions)"

echo "Done. Run: git status && git log --oneline -15"
