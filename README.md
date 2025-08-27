# blicket-text
Blicket text game

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with your Firebase credentials:
```bash
# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYour private key here\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your-service-account-email@your-project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your-client-id
FIREBASE_CLIENT_X509_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email%40your-project.iam.gserviceaccount.com
FIREBASE_DATABASE_URL=https://your-project-default-rtdb.firebaseio.com/
```

3. Run the application:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork this repository to your GitHub account

2. In your Streamlit Cloud dashboard:
   - Connect your GitHub repository
   - Set the main file path to `app.py`
   - Add the following secrets in the "Secrets" section:
     ```
     FIREBASE_PROJECT_ID = "your-project-id"
     FIREBASE_PRIVATE_KEY_ID = "your-private-key-id"
     FIREBASE_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\nYour private key here\n-----END PRIVATE KEY-----\n"
     FIREBASE_CLIENT_EMAIL = "your-service-account-email@your-project.iam.gserviceaccount.com"
     FIREBASE_CLIENT_ID = "your-client-id"
     FIREBASE_CLIENT_X509_CERT_URL = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email%40your-project.iam.gserviceaccount.com"
     FIREBASE_DATABASE_URL = "https://your-project-default-rtdb.firebaseio.com/"
     ```

3. Deploy!

## Firebase Setup

1. Create a Firebase project at https://console.firebase.google.com/
2. Enable Realtime Database
3. Create a service account and download the JSON file
4. Extract the credentials from the JSON file and add them to your `.env` file or Streamlit secrets

## Features

- Visual and text-only modes for blicket detection game
- Real-time state history tracking
- Firebase integration for data collection
- Responsive design with sidebar navigation
