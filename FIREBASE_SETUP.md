# 🔥 Firebase Setup Guide for Blicket Text Game

This guide will help you create a new Firebase project and configure it for your Blicket text game.

## Step 1: Create Firebase Project

1. **Go to Firebase Console**: [https://console.firebase.google.com/](https://console.firebase.google.com/)

2. **Create a new project**:
   - Click "Create a project" or "Add project"
   - Enter a project name (e.g., "blicket-text-game")
   - Choose whether to enable Google Analytics (optional for this project)
   - Click "Create project"

3. **Note your Project ID** - you'll see it in the project settings

## Step 2: Set up Realtime Database

1. **In your Firebase project dashboard**:
   - Click on "Realtime Database" in the left sidebar
   - Click "Create Database"
   - Choose "Start in test mode" (for development)
   - Select a location (choose one close to you)
   - Click "Done"

2. **Note your Database URL** - it will look like: `https://your-project-id-default-rtdb.firebaseio.com/`

## Step 3: Create Service Account

1. **Go to Project Settings**:
   - Click the gear icon ⚙️ next to "Project Overview"
   - Select "Project settings"

2. **Create Service Account**:
   - Go to the "Service accounts" tab
   - Click "Generate new private key"
   - Download the JSON file (keep it secure!)

## Step 4: Configure Your App

### Option A: Use the Setup Script (Recommended)

1. **Run the setup script**:
   ```bash
   source venv/bin/activate
   python setup_firebase.py
   ```

2. **Follow the prompts**:
   - Enter your Firebase Project ID
   - Enter your Database URL
   - Provide the path to your downloaded JSON file

3. **Test the connection**:
   ```bash
   python setup_firebase.py test
   ```

### Option B: Manual Configuration

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** with your Firebase credentials from the downloaded JSON file

## Step 5: Test Your Setup

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. **Visit**: http://localhost:8501

## Database Structure

Your Firebase Realtime Database will store data in this structure:
```
{
  "participant_id": {
    "config": { ... },
    "created_at": "timestamp",
    "status": "configured",
    "games": {
      "game_id": {
        "start_time": "timestamp",
        "events": [...],
        "binary_answers": {...},
        "qa_time": "timestamp",
        "round_config": {...},
        "round_number": 1
      }
    }
  }
}
```

## Security Rules (Optional)

For production, you might want to update your database rules:

```json
{
  "rules": {
    ".read": "auth != null",
    ".write": "auth != null"
  }
}
```

## Troubleshooting

- **Import errors**: Make sure virtual environment is activated
- **Firebase errors**: Check your .env file and credentials
- **Database errors**: Ensure Realtime Database is enabled
- **Permission errors**: Check your service account permissions

## Next Steps

Once Firebase is configured:
1. Your app will save participant data and game results
2. You can view the data in the Firebase Console
3. The app will work fully with data persistence

## Security Notes

- Never commit your `.env` file to version control
- Keep your service account JSON file secure
- Consider using Firebase Authentication for production
