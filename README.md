# Blicket Text Game

A text-based interactive experiment for understanding causal reasoning through the blicket detector paradigm. Participants interact with a machine that activates based on specific rules, and their learning process is tracked in real-time.

## Overview

This application presents a causal learning task where participants must discover which objects (blickets) cause a machine to activate under different rule conditions (conjunctive vs. disjunctive). The experiment consists of two phases:

1. **Comprehension Phase**: Practice round to familiarize participants with the interface
2. **Main Experiment**: Three rounds with different blicket configurations and rules

## Features

### Interactive Gameplay
- **Text-based interface** for placing/removing objects on the blicket machine
- **Step-by-step visualization** of machine state changes
- **Action history tracking** showing all player actions and resulting states
- **Limited steps per round** to encourage strategic exploration

### Data Collection

The application automatically collects comprehensive behavioral data for each participant:

#### Participant Data Structure
```json
{
  "config": {
    "participant_id": "unique_id",
    "start_time": "timestamp",
    "total_rounds": 3,
    "experiment_type": "conjunctive_or_disjunctive"
  },
  "comprehension": {
    "action_history": [...],
    "total_actions": 5,
    "total_steps_taken": 10
  },
  "main_game": {
    "round_1": {...},
    "round_1_progress": {...},
    "round_2": {...},
    "round_2_progress": {...},
    "round_3": {...},
    "round_3_progress": {...}
  }
}
```

#### Round Data Fields
Each round captures:
- **Actions**: All place/remove actions with timestamps
- **Blicket Classifications**: Yes/No responses for each object
- **User Chosen Blickets**: Indices of objects the participant identified as blickets
- **Rule Hypothesis**: Free-text description of the participant's hypothesis
- **Rule Type**: Selected rule classification (Conjunctive vs. Disjunctive)
- **Ground Truth**: True blickets and true rule for this round
- **Objects on Machine Before Q&A**: State of the machine when Q&A phase started
- **Time Metrics**: Start time, end time, and total duration
- **State History**: Complete machine state transitions throughout the round

### Question & Answer Phase

After each round, participants complete:
1. **Blicket Classification**: For each object, indicate whether it's a blicket
2. **Rule Hypothesis**: Write a free-text description of the hypothesized rule
3. **Rule Type Selection**: Choose between conjunctive (ALL blickets required) or disjunctive (ANY blicket activates)

## Setup

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Configure Firebase credentials:**

   Create a `.env` file in the root directory:
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

4. **Run the application:**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **In Streamlit Cloud dashboard:**
   - Connect your GitHub repository
   - Set the main file path to `app.py`
   - Add Firebase credentials as secrets in the "Secrets" section (see STREAMLIT_CLOUD_SECRETS.md for details)

3. **Deploy!**

## Firebase Setup

1. Create a Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable **Realtime Database** (not Firestore)
3. Create a service account:
   - Go to Project Settings → Service Accounts
   - Click "Generate New Private Key"
   - Download the JSON file
4. Configure security rules to require authentication (see SETUP.md for details)
5. Extract credentials from the JSON and add them to your environment variables

## Data Analysis

Data is stored in Firebase Realtime Database with the following structure:

```
participants/
  └── {participant_id}/
      ├── config/
      ├── comprehension/
      └── main_game/
          ├── round_1/
          ├── round_1_progress/
          ├── round_2/
          ├── round_2_progress/
          ├── round_3/
          └── round_3_progress/
```

Each round includes:
- Action history with timestamps
- User inferences (hypothesis text and rule type)
- Chosen blickets vs. ground truth
- Complete state transitions
- Time-to-completion metrics

## Technical Details

- **Framework**: Streamlit (Python web app framework)
- **Database**: Firebase Realtime Database
- **Language**: Python 3.9+
- **Key Dependencies**: streamlit, firebase-admin, numpy

## File Structure

```
blicket-text-game/
├── app.py                      # Main Streamlit application and participant flow
├── textual_blicket_game.py    # Game logic and interface
├── env/
│   └── blicket_text.py        # Environment logic for blicket machine
├── images/                     # UI assets
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── SETUP.md                    # Detailed setup instructions
└── STREAMLIT_CLOUD_SECRETS.md  # Cloud deployment guide
```

## License

See LICENSE file for details.

## Citation

If you use this application in your research, please cite appropriately.
