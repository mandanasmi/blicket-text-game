# Blicket Text Game

A text-based interactive experiment for understanding causal reasoning through the blicket detector paradigm. Participants interact with a machine that activates based on specific rules, and their learning process is tracked in real-time.

## Overview

This application presents a causal learning task where participants must discover which objects (blickets) cause a machine to activate under different rule conditions (conjunctive vs. disjunctive). The experiment consists of two phases:

1. **Comprehension Phase**: Practice round to familiarize participants with the interface
2. **Main Experiment**: Three rounds with different blicket configurations and rules

## Features

### Interactive Gameplay
- **Text-based interface** for placing/removing objects on the blicket machine
- **0-based object indexing**: Objects are labeled as "Object 0", "Object 1", "Object 2", "Object 3"
- **Step-by-step visualization** of machine state changes showing before/after states
- **Action history tracking** showing all player actions, timestamps, and resulting states
- **State history display** with visual indicators showing which objects are on the machine at each step
- **Machine activation feedback** showing whether the machine is ON or OFF after each action
- **No step limits** - participants can explore freely
- **Session state reset** between rounds and phases to ensure clean data collection

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
    "action_history": [
      {
        "action": "place",
        "object_index": 0,
        "timestamp": "...",
        "machine_state_before": false,
        "machine_state_after": false,
        "objects_on_machine": [0]
      }
    ],
    "total_actions": 5,
    "total_steps_taken": 10,
    "user_actions": ["place Object 0", "remove Object 0", ...]
  },
  "main_game": {
    "round_1_20240101_120000_123": {
      "round_id": "round_1_20240101_120000_123",
      "round_number": 1,
      "total_actions": 8,
      "action_history": [...],
      "state_history": [...],
      "blicket_classifications": {
        "object_0": "No",
        "object_1": "Yes",
        "object_2": "Yes",
        "object_3": "No"
      },
      "user_chosen_blickets": [1, 2],
      "rule_hypothesis": "Both objects need to be on the machine",
      "rule_type": "Conjunctive",
      "objects_on_machine_before_qa": [1, 2],
      "round_config": {
        "rule": "conjunctive",
        "blicket_indices": [0, 1],
        "num_objects": 4
      },
      "true_blicket_indices": [0, 1],
      "true_rule": "conjunctive",
      "start_time": "...",
      "end_time": "...",
      "total_time_seconds": 245.5,
      "phase": "main_experiment",
      "interface_type": "text"
    }
  }
}
```

#### Round Data Fields
Each round captures:
- **Round ID**: Unique timestamped identifier for the round
- **Actions**: All place/remove actions with timestamps and machine state changes
- **Action History**: Detailed history showing `machine_state_before`, `machine_state_after`, and `objects_on_machine` for each action
- **State History**: Complete machine state transitions throughout the round
- **Blicket Classifications**: Yes/No responses for each object (0-based indexing)
- **User Chosen Blickets**: 0-based indices of objects the participant identified as blickets
- **Rule Hypothesis**: Free-text description of the participant's hypothesis
- **Rule Type**: Selected rule classification (Conjunctive vs. Disjunctive)
- **Objects on Machine Before Q&A**: 0-based indices of objects on machine when Q&A phase started
- **Round Config**: The true configuration including `blicket_indices` (0-based) and `rule`
- **True Blicket Indices**: 0-based ground truth indices of actual blickets
- **True Rule**: Ground truth rule (conjunctive or disjunctive)
- **Time Metrics**: Start time, end time, and total duration in seconds
- **Phase**: Always "main_experiment" for main game rounds
- **Interface Type**: Always "text" for text-based interface

**Note**: Object indexing is 0-based throughout (Object 0, Object 1, Object 2, Object 3).

### Question & Answer Phase

After each round, participants complete:
1. **Blicket Classification**: For each object (Object 0, 1, 2, 3), indicate "Yes" or "No" - whether it's a blicket
2. **Rule Hypothesis**: Write a free-text description of the hypothesized rule in a text area
3. **Rule Type Selection**: Choose between:
   - **Conjunctive**: ALL blickets must be on the machine (AND rule)
   - **Disjunctive**: ANY blicket on the machine activates it (OR rule)

All responses are automatically saved and cannot be skipped.

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
{participant_id}/
  ├── config/
  ├── comprehension/
  │   └── action_history
  └── main_game/
      ├── round_1_20240101_120000_123/  (timestamped)
      ├── round_2_20240101_120530_456/  (timestamped)
      └── round_3_20240101_121100_789/  (timestamped)
```

**Important**: No intermediate `round_X_progress` entries are created - only final timestamped round data is saved.

Each round entry (keyed by timestamp) includes:
- Complete action history with timestamps and machine state changes
- All Q&A responses (blicket classifications, hypothesis, rule type)
- User-chosen blicket indices vs. ground truth blicket indices
- Complete state transitions showing machine activation for each action
- Objects on machine before Q&A phase
- Time-to-completion metrics (start, end, total duration)
- Round configuration with true blickets and true rule

### Data Integrity

- **No progress entries**: The app no longer creates `round_X_progress` entries, ensuring a clean database
- **Unique timestamps**: Each round has a unique timestamped ID (format: `round_N_YYYYMMDD_HHMMSS_mmm`)
- **Automatic saving**: All data is saved automatically when completing each round
- **Session preservation**: Comprehension phase data is preserved when transitioning to main game

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
