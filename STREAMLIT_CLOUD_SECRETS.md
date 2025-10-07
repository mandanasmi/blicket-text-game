# Streamlit Cloud Secrets Configuration

## How to Add Firebase Secrets to Streamlit Cloud

### Step 1: Go to Streamlit Cloud Dashboard
1. Visit https://share.streamlit.io/
2. Sign in with your GitHub account
3. Find your `blicket-text-game` app
4. Click on the app settings (gear icon)

### Step 2: Add Secrets
Click on "Secrets" in the left sidebar and add the following configuration:

```toml
[firebase]
project_id = "rational-cat-473121-k4"
private_key_id = "1805ef1d2c3a2dc33b8b83e994dae3a150800bf7"
private_key = "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQCe+TW+8bwIZDDl\nND3ikNcEiVbGwLjrcTXZeuTmNFsW2c5z/yLHaovoETtLhQ87bo2jLwOoN4yBrxsJ\nkq7yy146QcyFd07dPTIKCk1UXZbA/441bxSjv6mW9pVRyMC+R1qL/UCbT8UXQsYF\nymNCLD1dGiy3RmnQEoDgkNgLvencNz/INsRee1//eCPMkPYYZ/BB3+SY2CaOM4df\nRxmpKFGivzUznVB564cA56zvziSDDMdLlb5rl/KpgwGjj7jPFoniGAHN1KADEUOs\nagXJ6TGLYXFYKriHAoCFHUjhD1cEpmHVx2JxL5S8q7zTGvFiJ5VbfCq5drStbQGb\n8KqAXw6HAgMBAAECggEAEBcr+CECOzHeO79x6WtnmASeooysgDLRPytP3SJney9I\nf40kuV0baWh1FqESeEgppd6trBUTRP0IQOEhPAnsMPv/9h0BVSX12UukUBRjnIfr\n85SZWN70DLJLTXUtrRS5IXox/oZbGI5K+hhmEN3myB73b7AnfZv3LMOnBRPlMRjC\n9msBaj2pNJCqNgvtN+3HC478CrA4tQsoNi/xKMO6TgEJIPy6JrUGROeki4AGV95C\n9cVUHXh5+sflg01liXSxruKnx219gnz+jHEUpgE9RlRTjtTCcgVTMcp9GqiUu1Cx\nNTIDAklvKneUisUYbAwSBIzR775u9HAZMj9E8H/qoQKBgQDQeqjZcaXH3uFdgKKU\nDMlGpdE1b/cXggOtfpoJhDxHS1MKYNGH7bZvnBIC9IyRL1Lo6RDz/WbgODAPk0nk\nFFzkR/Cm8gGD0yE9wv4/nGyskA0e/fONDKDni7mocovG6Cvlke06V+Ib6wNPoH1b\nH20htE5Set2Qt/scMsnBVyDT9QKBgQDDNcGwrciLo+MiCxBfh5FSv+VuT3ZBvFSY\nN7xfAEAN+sX+Y6DMYJp53DF1lF79SB/0JmfMiPpk5+l/jPHE0mmPpWjpXc7n6FC5\nB9i39ZUXptOAskxSK2c2ASVNl7e2c3ASI4QCXn7dyC/MJgxarjQeuX3Dy1HRxgmu\n+ziku1lHCwKBgQCICuLQMjcqPCj4KQ7uaYGWlnH02wF016SqvGisORxUsbSYmyFg\nACECp+ehAhAQVb9WuXAUp5FQU5oZL9YR/a+4T7GcX2PZsBaLBQmAXQUVflLxnGon\n6su9DRKz9zt7KtoqTpVtcxfbe/qHJYVnxxQSu9DRKz9zt7KtoqTpVtcxfbe/qHJYVnxxQh4FUTwZV/8G7usb2yarbDWQKBgQDB\n0y5ubx8hB6kOtE2djM4Oi9sSnHOe8ZBNU4oGCgmP19+fpuySAZlgRfIV/SwT8PlJ\nQpjtzVRRvLNrolasRv/pUSPKEwrN2S3Niqz0ezN+OHbo4iBFtjLUvG59jJcs4ZH1\ncK4ybAEr1QkgeubpYu79UtA9CDRFRY134Jltd1g94QKBgQCbybuqjId+1Zxrc8S2\nLLKyQ0i44KJ0DXO3UFfGGaOzD6nQbxACLPmxsvtTDYyUMy8jbhOG19ZLMwarY3ml\nwThO4vpvaGqRcz5BmGeWupU1r2tK9AlJMZbF+H9d2FRO6D0XWa1ZaxbRqt7lXwjU\nFzzyT6YI7nUQEXcjxb9lEYABow==\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk-fbsvc@rational-cat-473121-k4.iam.gserviceaccount.com"
client_id = "108501216763566313542"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40rational-cat-473121-k4.iam.gserviceaccount.com"
database_url = "https://rational-cat-473121-k4-default-rtdb.firebaseio.com/"
```

### Step 3: Save and Redeploy
1. Click "Save" in the secrets configuration
2. The app will automatically redeploy
3. Wait 2-3 minutes for deployment to complete

### Step 4: Verify Data Collection
1. Visit your app: https://blicket-text-game.streamlit.app/
2. You should see "✅ Data saving enabled" instead of "⚠️ Data saving disabled"
3. Play a game and check your Firebase console: https://console.firebase.google.com/u/0/project/rational-cat-473121-k4/database/rational-cat-473121-k4-default-rtdb/data/~2F

## Troubleshooting

### If you still see "Data saving disabled":
1. Double-check that all secrets are copied exactly (especially the private key)
2. Make sure there are no extra spaces or characters
3. Ensure the app has been redeployed after saving secrets

### If you get Firebase errors:
1. Check that your Firebase project is active
2. Verify that the Realtime Database is enabled in Firebase console
3. Make sure the service account has proper permissions

