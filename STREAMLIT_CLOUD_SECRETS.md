# Streamlit Cloud Secrets Configuration

## How to Add Firebase Secrets to Streamlit Cloud

### Step 1: Go to Streamlit Cloud Dashboard
1. Visit https://share.streamlit.io/
2. Sign in with your GitHub account
3. Find your `nexiom-text-game` app (or the repository name)
4. Click on the app settings (gear icon)

### Step 2: Get Firebase Service Account Credentials
1. Go to [Firebase Console](https://console.firebase.google.com/u/0/project/nexiom-text-game)
2. Click on the gear icon (⚙️) next to "Project Overview"
3. Select "Project settings"
4. Go to the "Service accounts" tab
5. Click "Generate new private key"
6. Download the JSON file - this contains all the credentials you need

### Step 3: Add Secrets to Streamlit Cloud
Click on "Secrets" in the left sidebar and add the following configuration using the values from your downloaded JSON file:

```toml
[firebase]
project_id = "nexiom-text-game"
private_key_id = "YOUR_PRIVATE_KEY_ID_FROM_JSON"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk-XXXXX@nexiom-text-game.iam.gserviceaccount.com"
client_id = "YOUR_CLIENT_ID_FROM_JSON"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-XXXXX%40nexiom-text-game.iam.gserviceaccount.com"
database_url = "https://nexiom-text-game-default-rtdb.firebaseio.com/"
```

**Important Notes:**
- Replace all `YOUR_XXX` placeholders with actual values from your downloaded JSON file
- Replace `XXXXX` in the email addresses with your actual service account ID
- For the `private_key`, keep the `\n` characters exactly as shown (they represent newlines)

### Step 4: Save and Redeploy
1. Click "Save" in the secrets configuration
2. The app will automatically redeploy
3. Wait 2-3 minutes for deployment to complete

### Step 5: Verify Data Collection
1. Visit your app: https://nexiom-text-game.streamlit.app/
2. You should see "✅ Firebase connected - Data saving enabled" instead of "⚠️ Firebase not connected - Running in demo mode"
3. Play a game and check your Firebase console: https://console.firebase.google.com/u/0/project/nexiom-text-game/database/nexiom-text-game-default-rtdb/data/~2F

## Troubleshooting

### If you still see "Data saving disabled":
1. Double-check that all secrets are copied exactly (especially the private key)
2. Make sure there are no extra spaces or characters
3. Ensure the app has been redeployed after saving secrets

### If you get Firebase errors:
1. Check that your Firebase project is active
2. Verify that the Realtime Database is enabled in Firebase console
3. Make sure the service account has proper permissions

