import os
import json
import random
import datetime

import numpy as np
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv

import env.blicket_text as blicket_text
from textual_blicket_game import textual_blicket_game_page

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
        # Try Streamlit secrets first (for Cloud deployment)
        if hasattr(st, 'secrets') and hasattr(st.secrets, 'firebase') and 'firebase' in st.secrets:
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
            
        elif os.getenv("FIREBASE_PROJECT_ID"):  # Local development
            # Using local development Firebase credentials
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
            # Firebase initialized successfully
            
    except Exception as e:
        # Firebase initialization failed - app will run without data saving
        firebase_initialized = False
        # Firebase initialization failed

def create_new_game(seed=42, num_objects=4, num_blickets=2, rule="conjunctive"):
    """Initialize a fresh BlicketTextEnv and return it plus the first feedback."""
    random.seed(seed)
    np.random.seed(seed)
    env = blicket_text.BlicketTextEnv(
        num_objects=num_objects,
        num_blickets=num_blickets,
        init_prob=0.1,
        rule=rule,
        transition_noise=0.0,
        seed=seed,
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
    """Save game data to Firebase"""
    # Save game data
    if firebase_initialized and db_ref:
        try:
            participant_ref = db_ref.child(participant_id)
            games_ref = participant_ref.child('games')
            
            # Create a new game entry
            game_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            games_ref.child(game_id).set(game_data)
        except Exception as e:
            print(f"Failed to save game data: {e}")
    else:
        pass  # Firebase not available - game data not saved





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

    # Save current round data
    round_data = {
        "start_time": st.session_state.start_time.isoformat(),
        "events": [
            {"time": t.isoformat(), "entry": e}
            for t, e in zip(st.session_state.times, st.session_state.log)
        ],
        "binary_answers": binary_answers,
        "qa_time": qa_time.isoformat(),
        "round_config": st.session_state.round_configs[st.session_state.current_round],
        "round_number": st.session_state.current_round + 1
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
            rule=next_round_config['rule']
        )
        
        now = datetime.datetime.now()
        st.session_state.env = env
        st.session_state.start_time = now
        st.session_state.log = [first_obs]
        st.session_state.times = [now]
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
    st.session_state.phase = "intro"

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

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# 1) PARTICIPANT ID ENTRY SCREEN
if st.session_state.phase == "intro":
    # Show title
    st.title("🧙 Blicket Text Adventure")
    
    if not st.session_state.participant_id_entered:
        # Ask for Participant ID and start comprehension phase
        st.markdown(
            """
**Welcome to the Blicket Text Adventure!**

This is a text-only interface with 4 objects. 

**The study has two phases:**
1. **Comprehension Phase**: Learn the interface
2. **Main Experiment**: Actual experiment with data collection

Please enter your participant ID to begin.
"""
        )
        participant_id = st.text_input("Participant ID:", key="participant_id")
        if st.button("Start Comprehension Phase", type="primary") and participant_id.strip():
            st.session_state.current_participant_id = participant_id.strip()
            st.session_state.participant_id_entered = True
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
        ### Comprehension Phase
        
        This is the comprehension phase to help you understand the interface.
        
        **Instructions:**
        - You will see 4 objects (Object 1, Object 2, Object 3, Object 4)
        - Click on objects to place them on the blicket detector machine
        - Some objects are "blickets" that make the machine light up
        - Your goal is to figure out which objects are blickets and how the machine works
        - **You have exactly 5 actions** (placing or removing objects) to explore in this phase
        
        **The machine will show:**
        - 🟢 LIT = Machine is active
        - 🔴 NOT LIT = Machine is inactive
        
        When you're ready, click the button below to start the comprehension phase.
        """)
        
        if st.button("Start Comprehension Phase", type="primary"):
            # Create a simple practice configuration
            practice_config = {
                'num_objects': 4,
                'num_blickets': 2,
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
                rule=practice_config['rule']
            )
            
            st.session_state.env = env
            st.session_state.start_time = datetime.datetime.now()
            st.session_state.log = [first_obs]
            st.session_state.times = [datetime.datetime.now()]
            st.session_state.comprehension_completed = True
            st.session_state.phase = "practice_game"
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
    # Show title
    st.title("🧙 Blicket Text Adventure")
    
    st.markdown("## 🎉 Comprehension Phase Complete!")
    st.markdown(f"**Great job, {st.session_state.current_participant_id}!**")
    
    st.markdown("""
    ### Ready for the Main Experiment?
    
    Now that you've practiced with the interface, you're ready for the main experiment.
    
    **Main Experiment Structure:**
    - **3 rounds** total in the main experiment
    - **Same number of objects** (4 objects) in each round
    - **The rule may change** between rounds (conjunctive vs disjunctive)
    - **All data will be recorded** for research purposes
    - Each round is independent - what you learn in one round may not apply to the next
    
    
    Take your time to explore and understand each round. Click the button below when you're ready to start the main experiment.
    """)
    
    if st.button("Start Main Experiment", type="primary"):
        # Create random configuration for actual experiment (3 rounds with 4 objects)
        import random
        
        # Set random seed based on participant ID for reproducibility
        random.seed(hash(st.session_state.current_participant_id) % 2**32)
        
        num_rounds = 3
        round_configs = []
        
        for i in range(num_rounds):
            # Always use 4 objects
            num_objects = 4
            # Random number of blickets between 1 and 4
            num_blickets = random.randint(1, 4)
            # Random rule
            rule = random.choice(['conjunctive', 'disjunctive'])
            # Random initial probability
            init_prob = random.uniform(0.1, 0.3)
            # Random transition noise
            transition_noise = 0.0
            
            round_config = {
                'num_objects': num_objects,
                'num_blickets': num_blickets,
                'rule': rule,
                'init_prob': init_prob,
                'transition_noise': transition_noise,
                'horizon': 32  # Default step limit
            }
            round_configs.append(round_config)
        
        # Save configuration
        config = {
            'num_rounds': num_rounds,
            'user_selected_objects': 4,  # Fixed to 4 objects
            'rounds': round_configs,
            'interface_type': 'text'  # Fixed to text mode
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
            rule=round_config['rule']
        )
        
        now = datetime.datetime.now()
        st.session_state.env = env
        st.session_state.start_time = now
        st.session_state.log = [first_obs]
        st.session_state.times = [now]
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
    st.markdown(f"Thanks for playing, {st.session_state.current_participant_id}! All your responses have been saved to the database.")
    st.button("Start Over", on_click=reset_all)
