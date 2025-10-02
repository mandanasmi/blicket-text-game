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
from visual_blicket_game import visual_blicket_game_page

# Load environment variables
load_dotenv()

# Set text mode
os.environ['BLICKET_VISUAL_MODE'] = 'False'

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize Firebase using environment variables or Streamlit secrets
if not firebase_admin._apps:
    try:
        # Try Streamlit secrets first (for Cloud deployment)
        if hasattr(st, 'secrets') and 'firebase' in st.secrets:
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
        else:
            # Fall back to environment variables (for local development)
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
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url
        })
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {str(e)}")
        st.stop()

# Get database reference
db_ref = db.reference()

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
    participant_ref = db_ref.child(participant_id)
    participant_ref.set({
        'config': config,
        'created_at': datetime.datetime.now().isoformat(),
        'status': 'configured'
    })

def save_game_data(participant_id, game_data):
    """Save game data to Firebase"""
    participant_ref = db_ref.child(participant_id)
    games_ref = participant_ref.child('games')
    
    # Create a new game entry
    game_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    games_ref.child(game_id).set(game_data)

def reset_all():
    """Clears all session_state so we go back to the intro screen cleanly."""
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
if "num_objects_selected" not in st.session_state:
    st.session_state.num_objects_selected = None
if "participant_id_entered" not in st.session_state:
    st.session_state.participant_id_entered = False

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
st.title("📝 Blicket Text Adventure")
st.markdown("*Text-only version with no images - objects and machine status communicated through text*")

# 1) PARTICIPANT ID ENTRY SCREEN
if st.session_state.phase == "intro":
    if not st.session_state.participant_id_entered:
        # Step 1: Ask for Participant ID
        st.markdown(
            """
**Welcome to the Blicket Text Adventure!**

This version uses only text descriptions - no visual images of objects or the blicket detector.

Please enter your participant ID to begin.
"""
        )
        participant_id = st.text_input("Participant ID:", key="participant_id")
        if st.button("Continue") and participant_id.strip():
            st.session_state.current_participant_id = participant_id.strip()
            st.session_state.participant_id_entered = True
            st.rerun()
    else:
        # Step 2: Ask for number of objects
        st.markdown(
            f"""
**Hello {st.session_state.current_participant_id}!**

How many objects would you like to play with in each round?
"""
        )
        
        num_objects = st.selectbox(
            "Number of objects per round:",
            options=[3, 4, 5, 6, 7, 8],
            index=1,  # Default to 4 objects
            key="num_objects_selector"
        )
        
        st.info(f"📝 You will play with **{num_objects} objects** in each round. Objects will be represented as numbered buttons (Object 1, Object 2, etc.)")
        
        if st.button("Start Text Game"):
            st.session_state.num_objects_selected = num_objects
            
            # Create random configuration (3 rounds with user-specified number of objects)
            import random
            
            # Set random seed based on participant ID for reproducibility
            random.seed(hash(st.session_state.current_participant_id) % 2**32)
            
            num_rounds = 3
            round_configs = []
            
            for i in range(num_rounds):
                # Use user-specified number of objects
                num_objects = st.session_state.num_objects_selected
                # Random number of blickets between 1 and num_objects
                num_blickets = random.randint(1, num_objects)
                # Random rule
                rule = random.choice(['conjunctive', 'disjunctive'])
                # Random initial probability
                init_prob = random.uniform(0.1, 0.3)
                # Random transition noise
                transition_noise = 0.0 #random.uniform(0.0, 0.1)
                
                round_config = {
                    'num_objects': num_objects,
                    'num_blickets': num_blickets,
                    'rule': rule,
                    'init_prob': init_prob,
                    'transition_noise': transition_noise,
                    'horizon': 32  # Default step limit
                }
                round_configs.append(round_config)
            
            # Save configuration to Firebase
            config = {
                'num_rounds': num_rounds,
                'user_selected_objects': num_objects,
                'rounds': round_configs,
                'interface_type': 'text'
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
    
    st.stop()

# 2) GAME RUN
elif st.session_state.phase == "game":
    # Use visual blicket game interface with text mode enabled
    round_config = st.session_state.round_configs[st.session_state.current_round]
    
    # Pass the save_game_data function to avoid circular imports
    def save_text_game_data(participant_id, game_data):
        game_data['interface_type'] = 'text'
        save_game_data(participant_id, game_data)
    
    visual_blicket_game_page(
        st.session_state.current_participant_id,
        round_config,
        st.session_state.current_round,
        len(st.session_state.round_configs),
        save_text_game_data,
        use_visual_mode=False  # Force text mode
    )

# 3) NEXT ROUND HANDLING
elif st.session_state.phase == "next_round":
    # Move to next round
    st.session_state.current_round += 1
    st.session_state.phase = "game"
    st.rerun()

# 4) END-OF-GAME SCREEN
elif st.session_state.phase == "end":
    st.markdown("## 🎉 All done!")
    st.markdown(f"Thanks for playing the Text Adventure, {st.session_state.current_participant_id}! All your responses have been saved to the database.")
    st.button("Start Over", on_click=reset_all)
