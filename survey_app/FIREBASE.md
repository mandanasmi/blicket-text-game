# Firebase for the survey app

The survey app uses the **Firebase Admin SDK** (Python, server-side). It does **not** use the JavaScript/Web SDK. You must provide **service account** credentials and the **Realtime Database URL**, not the web config.

## Web SDK config (for reference only)

If you use a separate web or client app, this is the Firebase Web/JS config for project **nexiom-passive-participants**:

```js
const firebaseConfig = {
  apiKey: "AIzaSyDw_s9rRmIgfMnPNYhzrl7qn6Wro7b1Yxc",
  authDomain: "nexiom-passive-participants.firebaseapp.com",
  projectId: "nexiom-passive-participants",
  storageBucket: "nexiom-passive-participants.firebasestorage.app",
  messagingSenderId: "944204529110",
  appId: "1:944204529110:web:81f033c47976b53238f877",
  measurementId: "G-FXDZ979MFC"
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
