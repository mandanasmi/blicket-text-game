# Passive app

Participants see an action history and answer questions only; no comprehension phase and no object interaction.

**Flow:** Consent → intro (Prolific ID, demographics) → action history (one file per session, from `active_explore/analysis/action_histories/`; shown one step at a time via Next) → object identification (Is each object a Nexiom?) → rule description and rule type (Conjunctive / Disjunctive). History stays in the left sidebar; the main area shows one step at a time until they proceed to questions.

**Firebase:** Admin SDK (service account + Realtime Database URL). Use Streamlit secrets or `FIREBASE_*` env vars. See FIREBASE.md in this folder for the nexiom-passive-participants project and Admin vs Web SDK.

**Stored per participant:** `consent`, `demographics`, and under `game_data`: `action_history_text`, `num_objects`, `num_steps`, object answers, `rule_inference`, `rule_type`, `saved_at`, etc. `app_type` is `"survey_no_exploration"`.

## Run

From project root:

```bash
./run_passive_app.sh
```

or:

```bash
streamlit run passive_app/app.py --server.port 8504
```

http://localhost:8504

## Streamlit Cloud

Create an app from this repo, set **Main file path** to `passive_app/app.py`, add Firebase secrets (same TOML as main app; see root STREAMLIT_CLOUD_SECRETS.md). Redeploy. Use `passive_app/requirements.txt` for that app.

## Config

- `SURVEY_COMPLETION_CODE` – Prolific completion code
- `IRB_PROTOCOL_NUMBER` – optional
- Firebase: `.streamlit/secrets.toml` or Streamlit Cloud Secrets; or `FIREBASE_*` env vars. SECRETS_TOML.md in this folder has a template.
