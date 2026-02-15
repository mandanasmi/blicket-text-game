# Survey app (no exploration)

Collect human data without a comprehension phase or object interaction.

- **No comprehension phase** – consent then intro (participant ID, demographics) then straight to action history and questions.
- **Action history as text only** – participants upload a `.txt` file or paste text (one step per line; optional format: `action | Machine: ON/OFF`). Only the raw text is uploaded to Firebase.
- **Blicket questions and rule inference only** – no object interaction or exploration. Participants answer:
  - For each object: Is it a Nexiom? (Yes/No)
  - Rule description (free text)
  - Rule type: Conjunctive / Disjunctive

## Firebase

The app uses the **Firebase Admin SDK** (service account + Realtime Database URL). You can point it at the main app project or at a separate project (e.g. **nexiom-passive-participants**). Set credentials in Streamlit secrets or `FIREBASE_*` env vars. See **FIREBASE.md** in this folder for using the nexiom-passive-participants project and the difference between the Web SDK config and the Admin SDK credentials.

Per participant, data is stored under `participant_id/`:

- `consent` – consent given, timestamp, IRB protocol
- `demographics` – prolific_id, age, gender
- `survey` – survey response:
  - `action_history_text` – raw action history string
  - `num_objects`, `num_steps`
  - `blicket_classifications` – e.g. `{"object_0": "Yes", "object_1": "No", ...}`
  - `rule_hypothesis`, `rule_type`
  - `saved_at`, `session_timestamp`, `phase`, `app_type`

`app_type` is set to `"survey_no_exploration"` so you can filter this data from the main experiment.

## Run

From project root:

```bash
./run_passive_app.sh
```

Or:

```bash
streamlit run passive_app/app.py --server.port 8504
```

Open http://localhost:8504.

## Streamlit Cloud (e.g. blicket-text-game-passive.streamlit.app)

To serve this passive app at a URL like `https://blicket-text-game-passive.streamlit.app/`:

1. In [Streamlit Cloud](https://share.streamlit.io/), open the app that uses that URL.
2. In **Settings** → **General**, set **Main file path** to `passive_app/app.py`.
3. In **Secrets**, add the same Firebase TOML as the main app (see project root `STREAMLIT_CLOUD_SECRETS.md`).
4. Save and redeploy. The passive app will run at that URL.

Cloud uses the **entrypoint directory** for dependencies: with Main file path `passive_app/app.py`, it should use `passive_app/requirements.txt` (streamlit, firebase-admin, python-dotenv only). If the build still fails, ensure this `passive_app/requirements.txt` is committed and that the app’s Main file path is exactly `passive_app/app.py`.

## Config

- `SURVEY_COMPLETION_CODE` – completion code for Prolific (default: same as main app).
- `IRB_PROTOCOL_NUMBER` – optional.
- Firebase: use `.streamlit/secrets.toml` (local) or Streamlit Cloud **Secrets** (TOML format). See **SECRETS_TOML.md** in this folder for a copy-paste TOML template for the passive app (nexiom-passive-participants). You can also use `FIREBASE_*` env vars.
