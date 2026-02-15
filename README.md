# Nexiom Text Game

Text-based experiment for causal reasoning via the Nexiom machine: participants discover which objects (Nexioms) activate the machine under conjunctive or disjunctive rules.

**Phases:** (1) Comprehension – short practice round. (2) Main experiment – rounds with different Nexiom configurations. The main app currently uses **disjunctive** rules only.

**Interface:** Text-based place/remove and Test; objects are Object 0–3. Action and state history are shown. No step limit.

## Data collection

Per participant we store config, comprehension phase (action history, test actions), and main_game (per round: action history, state history, Nexiom classifications, user-chosen Nexioms, rule hypothesis, rule type, round config, true blickets/rule, timings). Object indices are 0-based. After each round participants answer: Nexiom Yes/No per object, free-text rule hypothesis, and Conjunctive vs Disjunctive.

## Setup

1. `pip install -r requirements.txt`
2. Optional: `python -m venv venv` and activate.
3. Create `.env` in the project root with Firebase credentials (FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL, FIREBASE_DATABASE_URL, etc.). See SETUP.md.
4. Run: `streamlit run app.py` or `./run_app_disjunctive.sh` → http://localhost:8501

Helper scripts: `run_app_disjunctive.sh` (port 8501), `run_app_conjunctive.sh` (port 8502).

## Passive app

The **passive app** (`passive_app/`) collects responses from participants who **do not** interact with the machine: they see an action history (assigned from files, shown one step at a time), then answer object identification and rule inference. No comprehension phase. Run: `./run_passive_app.sh` or `streamlit run passive_app/app.py --server.port 8504` → http://localhost:8504. See `passive_app/README.md` for Firebase, Cloud deployment, and action-history setup.

## Firebase and Cloud

Create a Firebase project, enable Realtime Database, add a service account, and put credentials in `.env` or Streamlit secrets. For Cloud deployment and secret format see STREAMLIT_CLOUD_SECRETS.md.

## Analysis

- **Basic:** Export Realtime DB as JSON, then `analysis/analyze_from_json.py` and (optionally) `analysis/visualizations/round1/visualize_results.py`. See `analysis/README.md`.
- **Round 7 / full pipeline:** `analysis/results/final_data_round7/prepare_results_human_data.py` and subfolders (accuracy, exploration_time, human_search_strategy, infogain_hypothesis, pnas, correlations).

## File structure (main pieces)

- `app.py` – main Streamlit app
- `textual_nexiom_game.py` – game logic and UI
- `run_app_disjunctive.sh`, `run_app_conjunctive.sh` – run main app
- `run_passive_app.sh` – run passive app
- `passive_app/app.py` – passive app (action history + Q&A)
- `env/blicket_text.py` – environment logic
- `analysis/` – analysis scripts and results (see analysis/README.md)

## License

See LICENSE.
