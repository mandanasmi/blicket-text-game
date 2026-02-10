# Streamlit Cloud Secrets (TOML)

Streamlit Cloud serves secrets to your app at runtime. Use **TOML format** in the dashboard under **Secrets**. Changes take around a minute to propagate. See **FIREBASE.md** for why the survey app needs the Admin SDK (service account + database URL), not the Web SDK config.

**Do not commit** your service account JSON or any file containing your `private_key`. Paste credentials only into Streamlit Cloud Secrets (or local `.streamlit/secrets.toml`, which is gitignored).

## Where to get the values

1. **Realtime Database** – In [Firebase Console](https://console.firebase.google.com/) for **nexiom-passive-participants**, enable **Realtime Database** and copy the database URL (e.g. `https://nexiom-passive-participants-default-rtdb.firebaseio.com`).

2. **Service account** – In the same project: Project settings (gear) → **Service accounts** → **Generate new private key**. Download the JSON. It contains:
   - `project_id`
   - `private_key_id`
   - `private_key`
   - `client_email`
   - `client_id`
   - `client_x509_cert_url`

## Survey app: Firebase (nexiom-passive-participants)

Paste this in the Secrets text area. Replace the placeholders with values from the downloaded service account JSON and the Realtime Database URL from the console.

```toml
[firebase]
project_id = "nexiom-passive-participants"
private_key_id = "from your downloaded JSON"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk-xxxxx@nexiom-passive-participants.iam.gserviceaccount.com"
client_id = "from your JSON"
client_x509_cert_url = "from your JSON"
database_url = "https://nexiom-passive-participants-default-rtdb.firebaseio.com"
```

- **project_id** – Use `"nexiom-passive-participants"` (or copy from the JSON).
- **private_key_id**, **client_id** – Copy from the service account JSON.
- **private_key** – Full PEM block from the JSON; keep newlines or use `\n` between lines.
- **client_email**, **client_x509_cert_url** – From the JSON; replace `xxxxx` with your service account ID in both if you type them by hand.
- **database_url** – From Firebase Console → Realtime Database; use the URL for your region if different (e.g. `...europe-west1.firebasedatabase.app`).

After saving, the survey app will read and write to the **nexiom-passive-participants** Realtime Database.

## Optional: completion code and IRB

To override defaults via Streamlit Cloud, you can set environment variables in **Settings → Environment variables** instead of in Secrets (the app reads `os.getenv("SURVEY_COMPLETION_CODE")` and `os.getenv("IRB_PROTOCOL_NUMBER")`). If you prefer Secrets:

```toml
# Add to the same Secrets TOML (top-level keys):
SURVEY_COMPLETION_CODE = "YOUR_PROLIFIC_COMPLETION_CODE"
IRB_PROTOCOL_NUMBER = "YOUR_IRB_NUMBER"
```

## Reference

- [Streamlit docs: Secrets management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- **FIREBASE.md** – Admin SDK vs Web SDK, and what the survey app needs.
- The app reads `st.secrets["firebase"]` and `os.getenv("FIREBASE_*")` (see `survey_app/app.py`).
