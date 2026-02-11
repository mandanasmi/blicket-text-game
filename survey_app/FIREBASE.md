# Firebase for the survey app

The survey app uses the **Firebase Admin SDK** (Python, server-side). It does **not** use the JavaScript/Web SDK. You must provide **service account** credentials and the **Realtime Database URL**, not the web config.

## Web SDK config (for reference only)

If you use a separate web or client app, this is the Firebase Web/JS config for project **nexiom-passive-participants**:

```js
const firebaseConfig = {
  apiKey: "AIzaSyDw_s9rRmIgfMnPNYhzrl7qn6Wro7b1Yxc",
  authDomain: "nexiom-passive-participants.firebaseapp.com",
  databaseURL: "https://nexiom-passive-participants-default-rtdb.firebaseio.com",
  projectId: "nexiom-passive-participants",
  storageBucket: "nexiom-passive-participants.firebasestorage.app",
  messagingSenderId: "944204529110",
  appId: "1:944204529110:web:fa13a704f446106638f877",
  measurementId: "G-28F5EWW5ZF"
};
```

Do **not** put this in the Python app. The survey app needs the Admin SDK credentials below.

## What the survey app needs (Admin SDK)

1. **Realtime Database** – In [Firebase Console](https://console.firebase.google.com/) for **nexiom-passive-participants**, enable **Realtime Database** and copy the database URL (e.g. `https://nexiom-passive-participants-default-rtdb.firebaseio.com`).

2. **Service account** – In the same project: Project settings (gear) → **Service accounts** → **Generate new private key**. Download the JSON. It contains:
   - `project_id`
   - `private_key_id`
   - `private_key`
   - `client_email`
   - `client_id`
   - `client_x509_cert_url`

3. **Secrets (Streamlit Cloud or local)** – In Streamlit secrets (or `.streamlit/secrets.toml` / env vars), set the same structure as the main app but with values for **nexiom-passive-participants**:

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

Use the actual values from the downloaded service account JSON and the Realtime Database URL from the console. Then the survey app will read and write to **nexiom-passive-participants** Realtime Database.

## Why don't I see data in Firebase?

1. **Wrong database_url** – `database_url` must be your **Realtime Database** URL (e.g. `https://nexiom-passive-participants-default-rtdb.firebaseio.com`). It must contain `firebaseio.com` or `firebasedatabase.app`. Do not use the Firebase Console overview page URL (`https://console.firebase.google.com/...`). In Firebase Console: **Build** -> **Realtime Database** -> copy the URL at the top.

3. **Check the right project** – Survey app writes to **nexiom-passive-participants** (if your secrets use that project). The main app uses **nexiom-text-game**. In [Firebase Console](https://console.firebase.google.com/) switch to the project that matches your secrets (`project_id` / `database_url`).

4. **Realtime Database, not Firestore** – Data is in **Realtime Database**. In the console: Build → Realtime Database → open the database that matches your `database_url`.

5. **Where data appears** – Under the root you see one key per participant (their Prolific ID). Each has: `consent`, `demographics`, `game_data` (and `created_at`, `status`, etc.). Consent and demographics are written after they click Continue on the intro; `game_data` is written only when they click **Submit responses** on the rule-type page. If they don’t finish the flow, `survey` won’t exist.

6. **Firebase not connected** – If the app shows a warning that Firebase is not connected, nothing is saved. Fix:
   - **Streamlit Cloud:** App → Manage app → Settings → Secrets. Add the full `[firebase]` TOML (see SECRETS_TOML.md). Redeploy.
   - **Local:** Create `.streamlit/secrets.toml` (gitignored) with the same `[firebase]` block. Restart the app.

7. **Database rules** – In Realtime Database → Rules, ensure writes are allowed for your setup (e.g. only authenticated/server credentials). Invalid rules can cause silent failures.
