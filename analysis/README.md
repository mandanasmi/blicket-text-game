# Analysis

Scripts for data from the main app (active) and the passive app.

## Quick start

1. Export Realtime Database as JSON from Firebase Console; save as `firebase_data.json` in `analysis/`.
2. Run:
   ```bash
   cd analysis
   python3 analyze_from_json.py
   ```
   Optional: `python3 generate_comprehensive_csv.py` (expects `firebase_data2.json` for a fuller CSV).
3. Visualizations: `python3 visualizations/round1/visualize_results.py` (uses the CSVs from step 2). For the round-7 pipeline use `results/final_data_round7/` (see below).

## What gets analyzed

Comprehension correctness, exploration (tests per session, by phase/round), and rule understanding (conjunctive vs disjunctive accuracy, blicket identification). Participants are filtered by Prolific-style IDs (24-char hex); test/demo IDs are excluded.

## Outputs (basic)

`analyze_from_json.py` writes CSVs (e.g. comprehension, exploration, rule understanding). The round-1 visualizer produces PNGs from those CSVs.

## Layout

- `analyze_from_json.py` – reads `firebase_data.json`, writes CSVs
- `generate_comprehensive_csv.py` – from `firebase_data2.json` (comprehension + main game rounds)
- `analyze_accuracy_comprehensive.py` – accuracy analysis
- `extract_passive_data.py` – passive data from `data_txt_history/` JSON
- `visualizations/round1/` – e.g. `visualize_results.py`
- `visualizations/round2/` – accuracy, prior experience
- `results/final_data_round7/` – round-7 human data: run `prepare_results_human_data.py` then use subfolders (accuracy, exploration_time, human_search_strategy, infogain_hypothesis, pnas, correlations)
- `llm_analysis/` – LLM vs human comparisons

## Round 7 pipeline

From `analysis/results/final_data_round7/` run `prepare_results_human_data.py`. Data format is defined by those scripts; the subfolders contain the rest of the pipeline.

## Dependencies

`pip install -r requirements.txt` (from `analysis/` or project root). Scripts use pandas, numpy, matplotlib, seaborn and others as needed.

## Problems

- **firebase_data.json not found** – Export the Realtime DB and save the file in `analysis/`.
- **No Prolific participants** – IDs should be 24-char hex; test-like IDs are filtered out.
- **Module not found** – Install with `pip install -r requirements.txt`.
