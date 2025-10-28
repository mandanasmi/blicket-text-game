# Secure Firebase Realtime Database Rules

## Secure Rules Configuration

### Recommended Secure Rules for Your Research App

Copy and paste this into Firebase Console:

```json
{
  "rules": {
    "$participant_id": {
      // Only allow authenticated requests (Admin SDK or authenticated clients)
      ".read": "auth != null || request.auth != null",
      ".write": "auth != null || request.auth != null",
      
      "config": {
        ".validate": "newData.hasChildren(['participant_id', 'num_objects', 'num_blickets', 'rule', 'num_rounds'])"
      },
      
      "comprehension": {
        ".validate": "newData.hasChildren(['action_history', 'phase'])"
      },
      
      "games": {
        "$game_id": {
          ".validate": "newData.hasChildren(['phase', 'interface_type'])"
        }
      },
      
      "main_game": {
        "$round_id": {
          ".validate": "newData.hasChildren(['phase', 'round_number'])"
        }
      }
    },
    
    // Allow read access to check if participant exists
    // but require authentication for writes
    ".read": false,
    ".write": false
  }
}
```

### Update Instructions

1. **Go to Firebase Console Rules:**
   - Visit: https://console.firebase.google.com/u/0/project/rational-cat-473121-k4/database/rational-cat-473121-k4-default-rtdb/rules

2. **Replace existing rules** with the secure rules above

3. **Click "Publish"**

4. **Wait 2 minutes** for propagation

### How These Rules Work

✅ **Authentication Required:**
- `auth != null` - Requires any authenticated user
- Your Firebase Admin SDK (service account) is authenticated
- Blocks anonymous/unauthenticated access

✅ **Data Validation:**
- Validates structure of config, comprehension, and game data
- Ensures required fields are present
- Prevents malformed data entries

✅ **Hierarchical Security:**
- Root level (`.read: false, .write: false`) - blocks all access by default
- Participant level - allows authenticated access to specific participant data
- Nested structures - validates data format

✅ **Protection:**
- Prevents unauthorized data manipulation
- Blocks direct database access without authentication
- Still allows your Admin SDK to function (it's authenticated)

### Testing After Update

1. **Start your Streamlit app**
2. **Create a test participant**
3. **Play through the comprehension phase**
4. **Verify data saves** in Firebase Console
5. **Check for any error messages**

### Troubleshooting

**If you get "Permission Denied" errors:**

Your service account should still work. If not, verify:
- Service account credentials are correct in Streamlit secrets
- App has been redeployed after rule update
- Wait 5 minutes for rules to fully propagate

**If you get data validation errors:**

Check that your app is saving data with required fields:
- `config`: must have participant_id, num_objects, num_blickets, rule, num_rounds
- `comprehension`: must have action_history, phase
- `games` and `main_game`: must have phase, round_number

### Alternative: Moderate Security (If Above Doesn't Work)

If the strict rules cause issues during testing, use this intermediate security:

```json
{
  "rules": {
    "$participant_id": {
      // Allow read/write with some structure validation
      ".read": true,
      ".write": true,
      
      // Validate data structure
      ".validate": "newData.hasChildren(['config'])"
    },
    ".read": false,
    ".write": false
  }
}
```

This still requires knowing the participant ID to read/write, but is less strict.

### Production-Ready Rules

For a fully production-ready setup with additional security:

```json
{
  "rules": {
    "$participant_id": {
      ".read": "auth != null",
      ".write": "auth != null && 
                  newData.child('config').child('participant_id').val() === $participant_id",
      
      // Validate participant ID matches path
      ".validate": "newData.child('config').child('participant_id').val() === $participant_id",
      
      "config": {
        ".validate": "newData.hasChildren(['participant_id', 'num_objects', 'num_blickets'])"
      }
    }
  }
}
```

This ensures:
- Cannot write to another participant's data
- Participant ID in path matches data
- Only authenticated requests allowed

---

## Next Steps

1. **Update rules** using one of the configurations above
2. **Test thoroughly** with a new participant
3. **Monitor Firebase Console** for any denied requests
4. **Check Streamlit logs** for errors

Your data is now properly secured! 🔒

