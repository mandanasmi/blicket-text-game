from cgi import print_arguments, print_environ
import os
import json
import random
import datetime

import numpy as np
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv

# Guard print against BrokenPipeError in Streamlit teardown
import builtins as _builtins

def _safe_print(*args, **kwargs):
    try:
        _builtins.print(*args, **kwargs)
    except BrokenPipeError:
        pass
    except Exception:
        # Swallow any unexpected stdout errors to avoid crashing the app
        pass

print = _safe_print

import env.blicket_text as blicket_text
from textual_blicket_game import textual_blicket_game_page

# Optional: IRB protocol number can be provided via environment
IRB_PROTOCOL_NUMBER = os.getenv("IRB_PROTOCOL_NUMBER", "")

# Load environment variables
load_dotenv()

# Firebase initialization

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize Firebase - with fallback for deployment issues
firebase_initialized = False
db_ref = None

if not firebase_admin._apps:
    try:
        print("🔍 Attempting Firebase initialization...")
        
        # Try Streamlit secrets first (for both local and cloud deployment)
        if hasattr(st, 'secrets') and hasattr(st.secrets, 'firebase') and 'firebase' in st.secrets:
            print("✅ Found Streamlit secrets - using secrets.toml")
            firebase_credentials = {
                "type": "service_account",
                "project_id": st.secrets["firebase"]["project_id"],
                "private_key_id": st.secrets["firebase"]["private_key_id"],
                "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
                "client_email": st.secrets["firebase"]["client_email"],
                "client_id": st.secrets["firebase"]["client_id"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
                "universe_domain": "googleapis.com"
            }
            database_url = st.secrets["firebase"]["database_url"]
            
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred, {'databaseURL': database_url})
            db_ref = db.reference()
            firebase_initialized = True
            print("✅ Firebase initialized successfully using Streamlit secrets")
            
        elif os.getenv("FIREBASE_PROJECT_ID"):  # Fallback to environment variables
            print("⚠️ Using environment variables as fallback")
            firebase_credentials = {
                "type": "service_account",
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
                "universe_domain": "googleapis.com"
            }
            database_url = os.getenv("FIREBASE_DATABASE_URL")
            
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred, {'databaseURL': database_url})
            db_ref = db.reference()
            firebase_initialized = True
            print("✅ Firebase initialized successfully using environment variables")
        else:
            print("❌ No Firebase credentials found in secrets or environment variables")
            firebase_initialized = False
            
    except Exception as e:
        # Firebase initialization failed - app will run without data saving
        firebase_initialized = False
        print(f"❌ Firebase initialization failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✅ Firebase already initialized")
    firebase_initialized = True
    db_ref = db.reference()

def create_new_game(seed=42, num_objects=4, num_blickets=2, rule="conjunctive", blicket_indices=None):
    """Initialize a fresh BlicketTextEnv and return it plus the first feedback."""
    random.seed(seed)
    np.random.seed(seed)
    env = blicket_text.BlicketTextEnv(
        num_objects=num_objects,
        num_blickets=num_blickets,
        init_prob=0.0,  # Start with all objects OFF the machine
        rule=rule,
        transition_noise=0.0,
        seed=seed,
        blicket_indices=blicket_indices
    )
    game_state = env.reset()
    return env, game_state["feedback"]

def save_participant_config(participant_id, config):
    """Save participant configuration to Firebase"""
    # Save participant config
    if firebase_initialized and db_ref:
        try:
            participant_ref = db_ref.child(participant_id)
            participant_ref.set({
                'config': config,
                'created_at': datetime.datetime.now().isoformat(),
                'status': 'configured'
            })
        except Exception as e:
            print(f"Failed to save participant config: {e}")
    else:
        pass  # Firebase not available - config not saved

def save_game_data(participant_id, game_data):
    """Save game data to Firebase with enhanced tracking"""
    # Save game data
    if firebase_initialized and db_ref:
        try:
            participant_ref = db_ref.child(participant_id)
            
            # Determine which key to use based on phase
            phase = game_data.get('phase', 'unknown')
            if phase == 'main_experiment':
                games_ref = participant_ref.child('main_game')
            else:
                games_ref = participant_ref.child('games')  # Keep comprehension phase data in 'games'
            
            # Create a new game entry with detailed timestamp
            now = datetime.datetime.now()
            game_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            
            # Debug: Print what we're about to save
            print(f"🔥 save_game_data called - Round: {game_data.get('round_number', 'unknown')}")
            print(f"🔥 rule_type in game_data: '{game_data.get('rule_type', 'MISSING')}'")
            print(f"🔥 rule_hypothesis in game_data: '{game_data.get('rule_hypothesis', 'MISSING')[:50] if game_data.get('rule_hypothesis') else 'EMPTY'}...'")
            print(f"🔥 blicket_classifications in game_data: {game_data.get('blicket_classifications', 'MISSING')}")
            
            # Enhance game_data with additional metadata
            enhanced_game_data = {
                **game_data,
                "saved_at": now.isoformat(),
                "game_id": game_id,
                "session_timestamp": now.timestamp()
            }
            
            # Debug: Verify rule_type is in enhanced data
            print(f"🔥 rule_type in enhanced_game_data: '{enhanced_game_data.get('rule_type', 'MISSING')}'")
            
            games_ref.child(game_id).set(enhanced_game_data)
            print(f"✅ Successfully saved {phase} data for {participant_id} - Game ID: {game_id}")
        except Exception as e:
            print(f"❌ Failed to save game data: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    else:
        print("⚠️ Firebase not available - game data not saved")

def save_intermediate_progress_app(participant_id, phase, round_number=None, total_rounds=None, action_count=0):
    """Save intermediate progress - update entry with action history based on phase"""
    if firebase_initialized and db_ref:
        try:
            # Skip intermediate progress saves for main_experiment
            if phase == "main_experiment" or phase == "main_experiment_start":
                print(f"⚠️ Skipping intermediate progress save for main_experiment")
                return
            
            participant_ref = db_ref.child(participant_id)
            
            # Only save comprehension phase progress
            progress_ref = participant_ref.child('comprehension')
            entry_id = "action_history"
            
            # Create or update the entry
            now = datetime.datetime.now()
            
            # Get existing data or create new
            existing_data = progress_ref.child(entry_id).get() or {}
            
            # Update with current action history
            updated_data = {
                **existing_data,
                "last_updated": now.isoformat(),
                "user_actions": st.session_state.get('user_actions', []) if hasattr(st, 'session_state') else [],
                "total_actions": action_count,
                "phase": phase,
                "round_number": round_number,
                "total_rounds": total_rounds
            }
            
            progress_ref.child(entry_id).set(updated_data)
            print(f"💾 {phase} progress updated for {participant_id} - Round {round_number} - {action_count} actions")
            
        except Exception as e:
            print(f"❌ Failed to save {phase} progress for {participant_id}: {e}")
    else:
        print("⚠️ Firebase not available - progress not saved")





def handle_enter():
    """Callback for processing each command during the game."""
    cmd = st.session_state.cmd.strip().lower()
    if not cmd:
        return
    now = datetime.datetime.now()
    # record user command
    st.session_state.log.append(f"```{cmd}```")
    st.session_state.times.append(now)

    # step the environment
    game_state, reward, done = st.session_state.env.step(cmd)
    feedback = game_state["feedback"]

    now2 = datetime.datetime.now()
    st.session_state.log.append(feedback)
    st.session_state.times.append(now2)
    st.session_state.cmd = ""

    if done:
        st.session_state.phase = "qa"

def submit_qa():
    """Callback when the user clicks 'Submit Q&A'—saves to Firebase and continues or ends."""
    qa_time = datetime.datetime.now()
    binary_answers = {
        question: st.session_state.get(f"qa_{i}", "No")
        for i, question in enumerate(BINARY_QUESTIONS)
    }

    # Calculate total time spent on this round
    total_time_seconds = (qa_time - st.session_state.start_time).total_seconds()
    
    # Extract user actions (commands) from the log
    user_actions = []
    action_timestamps = []
    for i, (time, entry) in enumerate(zip(st.session_state.times, st.session_state.log)):
        if entry.startswith("```") and entry.endswith("```"):
            # This is a user command
            command = entry[3:-3]  # Remove the backticks
            user_actions.append(command)
            action_timestamps.append(time.isoformat())

    # Generate unique round ID
    round_id = f"qa_round_{st.session_state.current_round + 1}_{qa_time.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    
    # Get current round config
    current_round_config = st.session_state.round_configs[st.session_state.current_round]
    
    # Save current round data with enhanced tracking
    round_data = {
        "round_id": round_id,  # Unique identifier for this round
        "start_time": st.session_state.start_time.isoformat(),
        "end_time": qa_time.isoformat(),
        "total_time_seconds": total_time_seconds,
        "events": [
            {"time": t.isoformat(), "entry": e, "event_type": "command" if e.startswith("```") else "feedback"}
            for t, e in zip(st.session_state.times, st.session_state.log)
        ],
        "user_actions": user_actions,
        "action_timestamps": action_timestamps,
        "total_actions": len(user_actions),
        "action_history_length": len(user_actions),  # Length of action history
        "binary_answers": binary_answers,
        "qa_time": qa_time.isoformat(),
        "round_config": current_round_config,
        "round_number": st.session_state.current_round + 1,
        "true_rule": current_round_config.get('rule', 'unknown'),  # True rule for this round
        "true_blicket_indices": current_round_config.get('blicket_indices', []),  # True blickets (0-based)
        "phase": "main_experiment",
        "interface_type": "text"
    }
    
    save_game_data(st.session_state.current_participant_id, round_data)
    
    # Check if there are more rounds
    if st.session_state.current_round + 1 < len(st.session_state.round_configs):
        # Move to next round
        st.session_state.current_round += 1
        next_round_config = st.session_state.round_configs[st.session_state.current_round]
        
        # Create new game for next round
        env, first_obs = create_new_game(
            seed=42 + st.session_state.current_round,  # Different seed for each round
            num_objects=next_round_config['num_objects'],
            num_blickets=next_round_config['num_blickets'],
            rule=next_round_config['rule'],
            blicket_indices=next_round_config.get('blicket_indices', None)
        )
        
        now = datetime.datetime.now()
        st.session_state.env = env
        st.session_state.start_time = now
        st.session_state.log = [first_obs]
        st.session_state.times = [now]
        
        # Reset action history for next round
        st.session_state.steps_taken = 0
        st.session_state.user_actions = []
        st.session_state.action_history = []
        st.session_state.state_history = []
        st.session_state.selected_objects = set()  # Ensure all buttons start gray
        
        # Clear any remaining game state variables
        st.session_state.pop("visual_game_state", None)
        st.session_state.pop("rule_hypothesis", None)
        st.session_state.pop("rule_type", None)
        
        # Clear blicket question answers
        for i in range(10):  # Clear up to 10 possible blicket questions
            st.session_state.pop(f"blicket_q_{i}", None)
        
        st.session_state.phase = "game"
        
        # Clear Q&A state
        for i in range(len(BINARY_QUESTIONS)):
            st.session_state.pop(f"qa_{i}", None)
    else:
        # All rounds completed
        st.session_state.phase = "end"

def reset_all():
    """Clears all session_state so we go back to the intro screen cleanly."""
    for i in range(len(BINARY_QUESTIONS)):
        st.session_state.pop(f"qa_{i}", None)
    st.session_state.phase = "intro"
    st.session_state.env = None
    st.session_state.start_time = None
    st.session_state.log = []
    st.session_state.times = []
    st.session_state.current_participant_id = ""
    st.session_state.current_round = 0
    st.session_state.round_configs = []
    st.session_state.num_objects_selected = None
    st.session_state.participant_id_entered = False

BINARY_QUESTIONS = [
    "Did you test each object at least once?",
    "Did you use the feedback from the last test before making a decision?",
    "Were you confident in your final hypothesis?"
]

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# SESSION-STATE INITIALIZATION
if "phase" not in st.session_state:
    st.session_state.phase = "consent"
if "consent" not in st.session_state:
    st.session_state.consent = None
if "consent_timestamp" not in st.session_state:
    st.session_state.consent_timestamp = None

if "env" not in st.session_state:
    st.session_state.env = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "log" not in st.session_state:
    st.session_state.log = []
if "times" not in st.session_state:
    st.session_state.times = []
if "current_participant_id" not in st.session_state:
    st.session_state.current_participant_id = ""
if "current_round" not in st.session_state:
    st.session_state.current_round = 0
if "round_configs" not in st.session_state:
    st.session_state.round_configs = []
if "participant_id_entered" not in st.session_state:
    st.session_state.participant_id_entered = False
if "comprehension_completed" not in st.session_state:
    st.session_state.comprehension_completed = False
if "interface_type" not in st.session_state:
    st.session_state.interface_type = "text"  # Fixed to text mode
if "round_hypothesis" not in st.session_state:
    st.session_state.round_hypothesis = ""  # Store hypothesis for current round (string)
if "round_rule_type" not in st.session_state:
    st.session_state.round_rule_type = ""  # Store rule type for current round (string)

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# 0) CONSENT SCREEN
if st.session_state.phase == "consent":
    # Add CSS styling for the Accept button (green)
    st.markdown("""
    <style>
    /* Green styling for Accept button in consent form */
    .stApp .stButton button[kind="primary"]:nth-of-type(1) {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
    }
    
    .stApp .stButton button[kind="primary"]:nth-of-type(1):hover {
        background-color: #218838 !important;
        border-color: #218838 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Research Consent")
    st.markdown("**Please read the consent information below carefully. Participation is voluntary.**")

    with st.expander("Key Information", expanded=True):
        st.markdown(
            """
            - You are being invited to participate in a research study. Participation is completely voluntary.
            - Purpose: to examine how adults infer and interpret cause and effect and how adults understand the thoughts and feelings of other people.
            - Sessions: 1–4 testing sessions (usually one), each ≤ 30 minutes. You will play interactive games; sometimes you may receive payment incentives based on your choices. You may view short clips, images, or music and answer related questions.
            - Risks: primarily the risk of breach of confidentiality.
            - Benefits: no direct benefits. Results may improve our understanding of causal reasoning and social cognition.
            """
        )

    st.markdown("### Purpose of the Study")
    st.markdown(
        """
        This study is conducted by Alison Gopnik (UC Berkeley) and research staff.
        It investigates how adults infer and interpret cause and effect and how adults understand the thoughts and feelings of other people.
        Participation is entirely voluntary.
        """
    )

    st.markdown("### Study Procedures")
    st.markdown(
        """
        Up to 4 testing sessions (usually one), each ≤ 30 minutes. Tasks may include causal learning, linguistic, imagination, categorization/association, and general cognitive tasks. You may be asked to make judgments, answer questions, observe events, and perform actions (e.g., grouping objects or activating machines). You can skip any question and withdraw at any time without penalty. Attention checks ensure data quality; failure may result in rejection and no compensation.
        """
    )

    st.markdown("### Benefits")
    st.markdown("While there is no direct benefit, you may enjoy the interactive displays and contribute to science.")

    st.markdown("### Risks/Discomforts")
    st.markdown("We do not expect foreseeable risks beyond a minimal risk of confidentiality breach; safeguards are in place to minimize this risk.")

    st.markdown("### Confidentiality")
    st.markdown(
        """
        Your identity will be separated from your data and a random code used for tracking. Data are kept indefinitely and stored securely (encrypted, restricted access). Identifiers may be removed for future research use without additional consent. Your personal information may be released if required by law. Authorized representatives (e.g., sponsors such as NSF, Mind Science Foundation, Princeton University, Defense Advanced Research) may review data for study oversight.
        """
    )

    st.markdown("### Costs of Study Participation")
    st.markdown("There are no costs associated with study participation.")

    st.markdown("### Compensation")
    st.markdown(
        """
        For your participation in our research, you will receive a maximum rate of \$8 per hour. Payment ranges from \$0.54 to \$0.67 for a 5-minute task and from \$3.25 to \$4.00 for a 30-minute task, depending on the time it takes to complete the type of task you've been assigned. For studies on Prolific, you will receive a minimum rate of \$6.50 per hour. For experiments with a differential bonus payment system you may have the opportunity to earn "points" that are worth up to 5 cents each, with a total bonus of no more than 30 cents paid on top of the flat fee paid for the task completion. Your online account will be credited directly.
        """
    )

    st.markdown("### Rights")
    st.markdown(
        """
        Participation is voluntary. You are free to withdraw your consent and discontinue participation at any time without penalty or loss of benefits to which you are otherwise entitled.
        """
    )

    st.markdown("### Questions")
    st.markdown(
        """
        If you have any questions, please contact the lab at gopniklab@berkeley.edu or the project lead, Eunice Yiu, at ey242@berkeley.edu.If you have questions regarding your treatment or rights as a participant in this research project, contact the Committee for the Protection of Human Subjects at the University of California, Berkeley at (510) 642-7461 or subjects@berkeley.edu. 
        If you have questions about the software or analysis, please contact Mandana Samiei, at mandanas.samiei@mail.mcgill.ca. 
        """
    )

    if IRB_PROTOCOL_NUMBER:
        st.markdown(f"**IRB Protocol Number:** {IRB_PROTOCOL_NUMBER}")

    st.markdown(
        "> By selecting the \"Accept\" button below, I acknowledge that I am 18 or older, have read this consent form, and I agree to take part in the research. If you do NOT agree to participate in this study, please click the \"Decline\" button below."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept", type="primary"):
            st.session_state.consent = True
            st.session_state.consent_timestamp = datetime.datetime.now().isoformat()
            st.session_state.phase = "intro"
            st.rerun()
    with col2:
        if st.button("Decline", type="secondary"):
            st.session_state.consent = False
            st.session_state.consent_timestamp = datetime.datetime.now().isoformat()
            st.session_state.phase = "no_consent"
            st.rerun()

    st.stop()

# 0b) NO CONSENT SCREEN
elif st.session_state.phase == "no_consent":
    st.title("Thanks for your response")
    st.markdown("## You did not consent. The study will now close.")
    st.stop()

# 1) PARTICIPANT ID ENTRY SCREEN
if st.session_state.phase == "intro":
    # Show title
    st.title("🧙 Blicket Text Adventure")
    
    # Show Firebase connection status
    if firebase_initialized:
        print("✅ Firebase connected - Data saving enabled")
    else:
        print("⚠️ Firebase not connected - Running in demo mode (data will not be saved)")
    
    # Add CSS styling for the Start Comprehension Phase button (blue)
    st.markdown("""
    <style>
    /* Blue styling for Start Comprehension Phase button in demographics section */
    button[kind="primary"] {
        background-color: #0d47a1 !important;
        border-color: #0d47a1 !important;
        color: white !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #1565c0 !important;
        border-color: #1565c0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if not st.session_state.participant_id_entered:
        # Ask for Prolific ID and demographics, then start comprehension phase
        st.markdown(
            """
**Welcome to the Blicket Text Adventure!**

This is a text-only interface with 4 objects. 

**The study has two phases:**
1. **Comprehension Phase**: Learn the interface
2. **Main Experiment**: Actual experiment with data collection

Please enter your Prolific ID to begin and provide your age and gender.
"""
        )
        participant_id = st.text_input("Prolific ID:", key="participant_id")
        col_a, col_b = st.columns(2)
        with col_a:
            age_options = list(range(18, 100))
            age = st.selectbox("Age:", age_options, index=0, key="participant_age")
        with col_b:
            gender = st.selectbox(
                "Gender:",
                [
                    "Prefer not to say",
                    "Female",
                    "Male",
                    "Non-binary",
                    "Other",
                ],
                index=0,
                key="participant_gender",
            )

        if st.button("Next Step", type="primary"):
            if not participant_id.strip():
                st.warning("Please enter your Prolific ID to continue.")
                st.stop()

            st.session_state.current_participant_id = participant_id.strip()
            st.session_state.participant_id_entered = True
            # Persist consent information alongside config as soon as ID is available
            if firebase_initialized and db_ref and st.session_state.consent:
                try:
                    participant_ref = db_ref.child(st.session_state.current_participant_id)
                    participant_ref.child('consent').set({
                        'given': True,
                        'timestamp': st.session_state.consent_timestamp,
                        'irb_protocol_number': IRB_PROTOCOL_NUMBER
                    })
                    participant_ref.child('demographics').set({
                        'prolific_id': st.session_state.current_participant_id,
                        'age': int(st.session_state.participant_age) if st.session_state.get('participant_age') is not None else None,
                        'gender': st.session_state.participant_gender,
                    })
                except Exception:
                    pass
            st.session_state.phase = "comprehension"
            st.rerun()
    
    st.stop()

# 2) COMPREHENSION PHASE
elif st.session_state.phase == "comprehension":
    # Show title
    st.title("🧙 Blicket Text Adventure")
    
    if not st.session_state.comprehension_completed:
        st.markdown("## 🧠 Comprehension Phase")
        st.markdown(f"**Hello {st.session_state.current_participant_id}!**")
        
        st.markdown("""
        This is the comprehension phase to help you understand the interface.

        **Instructions:**
        - You will see 4 objects. Click to place them on the machine.
        - You can place one or more objects on the machine and click "Test."
        - If the machine lights up, the combination works.
        - Your goal is to figure out which object(s) turn the machine on and how it works.
        - Your tests and outcomes will appear in the Test History panel on the left-hand side.

        When you're ready, click the button below to start the comprehension phase.
        """)
        
        if st.button("Start Comprehension Phase", type="primary"):
            # Create a simple practice configuration
            practice_config = {
                'num_objects': 4,
                'num_blickets': 2,
                'blicket_indices': [1, 2],  # Objects 1 and 2 are blickets in comprehension phase
                'rule': 'conjunctive',
                'init_prob': 0.2,
                'transition_noise': 0.0,
                'horizon': 5  # Practice round limited to 5 actions only
            }
            
            # Create practice game
            env, first_obs = create_new_game(
                seed=999,  # Fixed seed for practice
                num_objects=practice_config['num_objects'],
                num_blickets=practice_config['num_blickets'],
                rule=practice_config['rule'],
                blicket_indices=practice_config['blicket_indices']
            )
            
            st.session_state.env = env
            st.session_state.start_time = datetime.datetime.now()
            st.session_state.log = [first_obs]
            st.session_state.times = [datetime.datetime.now()]
            st.session_state.comprehension_completed = True
            st.session_state.phase = "practice_game"
            
            # Save intermediate progress - starting comprehension game
            save_intermediate_progress_app(st.session_state.current_participant_id, "comprehension_game_start", 1, 1, 0)
            
            st.rerun()
    
    st.stop()

# 3) PRACTICE GAME
elif st.session_state.phase == "practice_game":
    # Show title
    st.title("🧙 Blicket Text Adventure")
    
    st.markdown("## Comprehension Phase - Round 1")
    st.markdown("**This is the comprehension phase to help you understand the interface.**")
    
    # Create a simple practice configuration
    practice_config = {
        'num_objects': 4,
        'num_blickets': 2,
        'blicket_indices': [1, 2],  # Objects 1 and 2 are blickets in comprehension phase
        'rule': 'conjunctive',
        'init_prob': 0.2,
        'transition_noise': 0.0,
        'horizon': 5
    }
    
    # Use the visual game page with data saving for comprehension phase
    def comprehension_save_func(participant_id, game_data):
        # Add phase identifier to distinguish comprehension data
        game_data['phase'] = 'comprehension'
        game_data['interface_type'] = st.session_state.interface_type
        save_game_data(participant_id, game_data)
    
    textual_blicket_game_page(
        st.session_state.current_participant_id,
        practice_config,
        0,  # Single practice round
        1,  # Total rounds = 1
        comprehension_save_func,
        use_visual_mode=False,
        is_practice=True
    )

# 4) PRACTICE COMPLETION
elif st.session_state.phase == "practice_complete":
    # Add CSS styling for the Start Main Experiment button (blue)
    st.markdown("""
    <style>
    /* Blue styling for Start Main Experiment button */
    .stApp .stButton button[kind="primary"] {
        background-color: #0d47a1 !important;
        border-color: #0d47a1 !important;
    }
    
    .stApp .stButton button[kind="primary"]:hover {
        background-color: #1565c0 !important;
        border-color: #1565c0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show title
    st.title("🧙 Blicket Text Adventure")
    
    st.markdown("## Comprehension Phase Complete!")
    st.markdown(f"**Great job, {st.session_state.current_participant_id}!**")
    
    # Progress indicator removed as requested
        
    if st.button("Start Main Experiment", type="primary", use_container_width=True):
        # Answer submitted (no need to validate - they can answer anything)
        # Move to main experiment
        # Create random configuration for actual experiment (3 rounds with 4 objects)
        import random
        
        # Set random seed based on participant ID for reproducibility
        random.seed(hash(st.session_state.current_participant_id) % 2**32)
        
        num_rounds = 3
        round_configs = []
        
        # Initialize rule generation
        current_rule = random.choice(['conjunctive', 'disjunctive'])
        rule_change_probability = 0.4  # 40% chance to change rule each round
        
        # Define diverse blicket combinations (0-based object indices 0-3)
        blicket_combinations = [
            [0, 1],  # Objects 0, 1
            [1, 2],  # Objects 1, 2
            [0, 2],  # Objects 0, 2
            [2, 3],  # Objects 2, 3
            [0, 3],  # Objects 0, 3
            [1, 3],  # Objects 1, 3
        ]
        
        # Shuffle combinations for variety
        random.shuffle(blicket_combinations)
        
        for i in range(num_rounds):
            # Always use 4 objects
            num_objects = 4
            # Select diverse blicket combination
            blicket_indices = blicket_combinations[i % len(blicket_combinations)]
            num_blickets = len(blicket_indices)
            
            # Rule logic: sometimes change, sometimes stay the same
            if i == 0:
                # First round: use initial rule
                rule = current_rule
            else:
                # Subsequent rounds: decide whether to change
                if random.random() < rule_change_probability:
                    # Change the rule
                    available_rules = ['conjunctive', 'disjunctive']
                    available_rules.remove(current_rule)
                    rule = random.choice(available_rules)
                    current_rule = rule
                else:
                    # Keep the same rule
                    rule = current_rule
            # Random initial probability
            init_prob = random.uniform(0.1, 0.3)
            # Random transition noise
            transition_noise = 0.0
            
            round_config = {
                'num_objects': num_objects,
                'num_blickets': num_blickets,
                'blicket_indices': blicket_indices,  # Specific objects that are blickets
                'rule': rule,
                'init_prob': init_prob,
                'transition_noise': transition_noise,
                'horizon': 16  # Default step limit
            }
            round_configs.append(round_config)
        
        # Save configuration
        config = {
            'num_rounds': num_rounds,
            'user_selected_objects': 4,  # Fixed to 4 objects
            'rounds': round_configs,
            'interface_type': 'text',  # Fixed to text mode
            'demographics': {
                'prolific_id': st.session_state.get('current_participant_id', ''),
                'age': int(st.session_state.participant_age) if 'participant_age' in st.session_state else None,
                'gender': st.session_state.get('participant_gender', 'Prefer not to say'),
            }
        }
        save_participant_config(st.session_state.current_participant_id, config)
        
        # Initialize first round
        st.session_state.current_round = 0
        st.session_state.round_configs = round_configs
        round_config = round_configs[0]
        env, first_obs = create_new_game(
            seed=42,
            num_objects=round_config['num_objects'],
            num_blickets=round_config['num_blickets'],
            rule=round_config['rule'],
            blicket_indices=round_config.get('blicket_indices', None)
        )
        
        now = datetime.datetime.now()
        st.session_state.env = env
        st.session_state.start_time = now
        st.session_state.log = [first_obs]
        st.session_state.times = [now]
        
        # Save final comprehension phase data before clearing session state
        save_intermediate_progress_app(st.session_state.current_participant_id, "comprehension", 0, 1, len(st.session_state.get('user_actions', [])))
        
        # Reset action history for main game (clear comprehension phase history)
        st.session_state.steps_taken = 0
        st.session_state.user_actions = []
        st.session_state.action_history = []
        st.session_state.state_history = []
        st.session_state.selected_objects = set()  # Ensure all buttons start gray
        
        # Clear any remaining game state variables
        st.session_state.pop("visual_game_state", None)
        st.session_state.pop("rule_hypothesis", None)
        st.session_state.pop("rule_type", None)
        
        # Clear blicket question answers
        for i in range(10):  # Clear up to 10 possible blicket questions
            st.session_state.pop(f"blicket_q_{i}", None)
        
        # Save intermediate progress - starting main experiment
        save_intermediate_progress_app(st.session_state.current_participant_id, "main_experiment_start", 1, num_rounds, 0)
        
        st.session_state.phase = "game"
        st.rerun()

# 5) MAIN GAME RUN
elif st.session_state.phase == "game":
    # Show title
    st.title("🧙 Blicket Text Adventure")
    
    # Use visual blicket game interface
    round_config = st.session_state.round_configs[st.session_state.current_round]
    
    # Pass the save_game_data function to avoid circular imports
    def save_visual_game_data(participant_id, game_data):
        game_data['phase'] = 'main_experiment'
        game_data['interface_type'] = st.session_state.interface_type
        save_game_data(participant_id, game_data)
    
    # Always use text mode
    use_visual_mode = False
    
    textual_blicket_game_page(
        st.session_state.current_participant_id,
        round_config,
        st.session_state.current_round,
        len(st.session_state.round_configs),
        save_visual_game_data,
        use_visual_mode=use_visual_mode
    )

# 3) NEXT ROUND HANDLING
elif st.session_state.phase == "next_round":
    # Move to next round
    st.session_state.current_round += 1
    
    # Capture hypothesis and rule from the previous round before clearing
    # These are set by the textual_blicket_game_page when user completes Q&A
    previous_hypothesis = st.session_state.get("rule_hypothesis", "")
    previous_rule_type = st.session_state.get("rule_type", "")
    
    print(f"💾 Captured for round {st.session_state.current_round}:")
    print(f"   - Hypothesis: {previous_hypothesis[:50] if previous_hypothesis else 'EMPTY'}...")
    print(f"   - Rule Type: {previous_rule_type}")
    
    # Reset action history for next round
    st.session_state.steps_taken = 0
    st.session_state.user_actions = []
    st.session_state.action_history = []
    st.session_state.state_history = []
    st.session_state.selected_objects = set()  # Ensure all buttons start gray
    
    # Reset hypothesis and rule for next round (they're strings, not lists)
    st.session_state.round_hypothesis = ""
    st.session_state.round_rule_type = ""
    
    # Clear Q&A variables for next round
    st.session_state.pop("rule_hypothesis", None)
    st.session_state.pop("rule_type", None)
    
    st.session_state.phase = "game"
    st.rerun()

# 4) Q&A (keeping for compatibility but not used in visual version)
elif st.session_state.phase == "qa":
    st.markdown(f"## 📝 Round {st.session_state.current_round + 1} Q&A")
    for i, question in enumerate(BINARY_QUESTIONS):
        st.radio(question, ("Yes", "No"), key=f"qa_{i}")
    
    if st.session_state.current_round + 1 < len(st.session_state.round_configs):
        st.button("Submit & Continue to Next Round", on_click=submit_qa)
    else:
        st.button("Submit & Finish", on_click=submit_qa)

# 5) END-OF-GAME SCREEN
elif st.session_state.phase == "end":
    st.markdown("## 🎉 All done!")
    st.markdown(f"Thanks for playing, {st.session_state.current_participant_id}!")
    
    # Progress indicator removed as requested
    
    st.markdown("""
    ### 🎯 Experiment Complete!
    
    
    Thank you for participating in our blicket research study!
    """)
    
    st.button("Start Over", on_click=reset_all)
