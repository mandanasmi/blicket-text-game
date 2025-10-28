# Firebase Realtime Database Security Rules

## Problem
Your Firebase Realtime Database was in "Test Mode" which:
- Allows read/write access to anyone
- Auto-expires after 30 days (expiring TODAY!)
- After expiration, ALL requests are denied

## Solution: Update Security Rules

### Option 1: Authenticated Access (Recommended)
Only allow authenticated users to access the database.

**Go to Firebase Console:**
1. Visit: https://console.firebase.google.com/u/0/project/rational-cat-473121-k4/database/rational-cat-473121-k4-default-rtdb/rules
2. Replace the rules with:

```json
{
  "rules": {
    ".read": "auth != null",
    ".write": "auth != null"
  }
}
```

### Option 2: App-Based Access (For Development/Research)
Allow specific access based on your app's needs while protecting data.

```json
{
  "rules": {
    ".read": true,
    ".write": true
  }
}
```

**⚠️ WARNING:** This allows anyone to read/write. Only use this if:
- You're in early development
- Your database doesn't contain sensitive data
- You'll implement proper security later

### Option 3: Time-Limited Testing (Temporary Fix)
Extend your test mode for another 30 days (one-time option).

**Go to Firebase Console:**
1. Visit: https://console.firebase.google.com/u/0/project/rational-cat-473121-k4/database/rational-cat-473121-k4-default-rtdb/rules
2. Click "Get Started" or "Extend Test Mode"

---

## Recommended Action for Your App

Since your app uses **Firebase Admin SDK** (service account) and not client authentication, use **Option 2** for now:

1. Go to: https://console.firebase.google.com/u/0/project/rational-cat-473121-k4/database/rational-cat-473121-k4-default-rtdb/rules
2. Click on "Realtime Database" → "Rules" tab
3. Replace the rules with:

```json
{
  "rules": {
    ".read": true,
    ".write": true
  }
}
```

4. Click "Publish"

### Why Option 2 for Your App?

- Your app uses Firebase Admin SDK (service account authentication)
- The Admin SDK bypasses security rules when used properly
- However, the auto-expiry of test mode blocks ALL access
- Setting read/write to `true` prevents the auto-expiry block

### For Better Security Later:

Consider implementing rate limiting and data validation:

```json
{
  "rules": {
    "$participant_id": {
      ".read": true,
      ".write": true,
      ".validate": "newData.hasChildren(['config', 'games'])"
    }
  }
}
```

---

## After Updating Rules

1. **Wait 2-5 minutes** for rules to propagate
2. **Test your app** - the errors should be resolved
3. **Monitor Firebase Console** for any access issues

## Verify It Works

1. Visit: https://console.firebase.google.com/u/0/project/rational-cat-473121-k4/database/rational-cat-473121-k4-default-rtdb/data
2. Try to access data - you should see data without errors
3. Test your Streamlit app - button clicks should work now

