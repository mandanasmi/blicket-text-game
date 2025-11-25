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

# Ensure app uses the browser width by default
st.set_page_config(layout="wide")

# Guard print against BrokenPipeError in Streamlit teardown
import builtins as _builtins

def _safe_print(*args, **kwargs):
    try:
        _builtins.print(*args, **kwargs)
    except BrokenPipeError:
        pass
    except Exception:
        pass

print = _safe_print

import env.blicket_text as blicket_text
from textual_blicket_game import textual_blicket_game_page

IRB_PROTOCOL_NUMBER = os.getenv("IRB_PROTOCOL_NUMBER", "")

load_dotenv()

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

firebase_initialized = False
db_ref = None

if not firebase_admin._apps:
    try:
        print("🔍 Attempting Firebase initialization...")

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

        elif os.getenv("FIREBASE_PROJECT_ID"):
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
        init_prob=0.0,
        rule=rule,
        transition_noise=0.0,
        seed=seed,
        blicket_indices=blicket_indices
    )
    game_state = env.reset()
    return env, game_state["feedback"]


def save_participant_config(participant_id, config):
    """Save participant configuration to Firebase"""
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


def save_game_data(participant_id, game_data):
    """Save game data to Firebase with enhanced tracking"""
    if firebase_initialized and db_ref:
        try:
            participant_ref = db_ref.child(participant_id)

            phase = game_data.get('phase', 'unknown')
            if phase == 'main_experiment':
                games_ref = participant_ref.child('main_game')
            else:
                games_ref = participant_ref.child('games')

            now = datetime.datetime.now()
            game_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]

            enhanced_game_data = {
                **game_data,
                "saved_at": now.isoformat(),
                "game_id": game_id,
                "session_timestamp": now.timestamp()
            }

            games_ref.child(game_id).set(enhanced_game_data)
            print(f"✅ Successfully saved {phase} data for {participant_id} - Game ID: {game_id}")
        except Exception as e:
            print(f"❌ Failed to save game data: {e}")
    else:
        print("⚠️ Firebase not available - game data not saved")


def save_intermediate_progress_app(participant_id, phase, round_number=None, total_rounds=None, action_count=0):
    """Save intermediate progress - comprehension only"""
    if firebase_initialized and db_ref:
        try:
            if phase in ("main_experiment", "main_experiment_start"):
                print(f"⚠️ Skipping intermediate progress save for main_experiment")
                return

            participant_ref = db_ref.child(participant_id)
            progress_ref = participant_ref.child('comprehension')
            entry_id = "action_history"

            now = datetime.datetime.now()
            existing_data = progress_ref.child(entry_id).get() or {}

            updated_data = {
                **existing_data,
                "last_updated": now.isoformat(),
                "user_actions": st.session_state.get('user_actions', []),
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

    st.session_state.log.append(f"```{cmd}```")
    st.session_state.times.append(now)

    game_state, reward, done = st.session_state.env.step(cmd)
    feedback = game_state["feedback"]

    now2 = datetime.datetime.now()
    st.session_state.log.append(feedback)
    st.session_state.times.append(now2)
    st.session_state.cmd = ""

    if done:
        st.session_state.phase = "qa"


def submit_qa():
    """Callback when user clicks 'Submit Q&A' (legacy text QA)."""
    qa_time = datetime.datetime.now()
    binary_answers = {
        question: st.session_state.get(f"qa_{i}", "No")
        for i, question in enumerate(BINARY_QUESTIONS)
    }

    total_time_seconds = (qa_time - st.session_state.start_time).total_seconds()

    user_actions = []
    action_timestamps = []
    for time, entry in zip(st.session_state.times, st.session_state.log):
        if entry.startswith("```") and entry.endswith("```"):
            command = entry[3:-3]
            user_actions.append(command)
            action_timestamps.append(time.isoformat())

    round_id = f"qa_round_{st.session_state.current_round + 1}_{qa_time.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    current_round_config = st.session_state.round_configs[st.session_state.current_round]

    round_data = {
        "round_id": round_id,
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
        "action_history_length": len(user_actions),
        "binary_answers": binary_answers,
        "qa_time": qa_time.isoformat(),
        "round_config": current_round_config,
        "round_number": st.session_state.current_round + 1,
        "true_rule": current_round_config.get('rule', 'unknown'),
        "true_blicket_indices": current_round_config.get('blicket_indices', []),
        "phase": "main_experiment",
        "interface_type": "text"
    }

    save_game_data(st.session_state.current_participant_id, round_data)

    if st.session_state.current_round + 1 < len(st.session_state.round_configs):
        st.session_state.current_round += 1
        next_round_config = st.session_state.round_configs[st.session_state.current_round]

        env, first_obs = create_new_game(
            seed=42 + st.session_state.current_round,
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

        st.session_state.steps_taken = 0
        st.session_state.user_actions = []
        st.session_state.action_history = []
        st.session_state.state_history = []
        st.session_state.selected_objects = set()

        st.session_state.pop("visual_game_state", None)
        st.session_state.pop("rule_hypothesis", None)
        st.session_state.pop("rule_type", None)
        for i in range(10):
            st.session_state.pop(f"blicket_q_{i}", None)

        st.session_state.phase = "game"
        for i in range(len(BINARY_QUESTIONS)):
            st.session_state.pop(f"qa_{i}", None)
    else:
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
    st.session_state.interface_type = "text"
if "round_hypothesis" not in st.session_state:
    st.session_state.round_hypothesis = ""
if "round_rule_type" not in st.session_state:
    st.session_state.round_rule_type = ""

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Global CSS for desktop screens
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stMain"] {
    width: 100% !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    padding: 0 !important;
    margin: 0 !important;
    position: relative !important;
}

body {
    overflow-x: hidden !important;
    background: #cfcfcf !important;
}

[data-testid="stMarkdown"] h1,
h1:first-of-type {
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
}


/* Hide right sidebar on phone screens only */
@media (max-width: 576px) {
    [data-testid="block-container"]::after {
        display: none !important;
    }
}

[data-testid="block-container"],
.block-container {
    width: clamp(360px, 44vw, 840px) !important;
    max-width: clamp(360px, 44vw, 840px) !important;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 2.5rem !important;
    margin-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    background-color: #ececec !important;
    border-radius: 26px !important;
    min-height: calc(100vh - 6rem) !important;
    overflow: visible !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.08) !important;
    box-sizing: border-box !important;
    position: relative !important;
}

@media (max-width: 1200px) {
    [data-testid="block-container"],
    .block-container {
        width: min(620px, calc(100vw - 2rem)) !important;
        max-width: min(620px, calc(100vw - 2rem)) !important;
        margin-right: auto !important;
    }
}

.object-grid-wrapper {
    max-width: 440px;
    margin: 0 auto;
}

.comprehension-layout {
    max-width: 720px;
    margin: 0 auto !important;
}

.comprehension-layout [data-testid="column"] {
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
    margin: 0.5rem 0 !important;
}

.comprehension-layout [data-testid="column"] > div {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    width: 100%;
}

.comprehension-layout [data-testid="stButton"] button {
    display: inline-flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 200px !important;
    min-width: 200px !important;
    border-radius: 18px !important;
    border: 2px solid #0d47a1 !important;
    box-shadow: 0 4px 10px rgba(13,71,161,0.25) !important;
    background-color: #f5f5f5 !important;
    color: #333 !important;
    font-weight: 700 !important;
    font-size: clamp(14px, 1.2vw, 18px) !important;
    padding: clamp(4px, 0.6vw, 8px) clamp(10px, 1.2vw, 18px) !important;
    margin: 0 auto !important;
    transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    cursor: pointer !important;
}

.comprehension-layout [data-testid="stButton"] button:hover {
    background-color: #e0f0ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 12px rgba(13,71,161,0.35) !important;
}

.object-grid-wrapper [data-testid="column"] {
    padding-left: 0.2rem !important;
    padding-right: 0.2rem !important;
    margin: 0.1rem !important;
}

.object-grid-wrapper [data-testid="stButton"] {
    margin-bottom: 0.25rem !important;
}

section[data-testid="stSidebar"] {
    min-width: 300px !important;
    width: 320px !important;
    max-width: 320px !important;
    max-height: 100vh !important;
    overflow-y: auto !important;
}

button, .stButton > button {
    font-size: 1.15rem !important;
    min-height: 2.4rem !important;
    border-radius: 0.5rem !important;
}
img { max-width: 100% !important; height: auto !important; }

button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondary"] {
    border-radius: 999px !important;
    border: 2px solid #0d47a1 !important;
    background: linear-gradient(135deg, #1e88e5 0%, #0d47a1 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    box-shadow: 0 6px 14px rgba(13, 71, 161, 0.35) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
    cursor: pointer !important;
}
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(13, 71, 161, 0.45) !important;
    font-size: 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# 0) CONSENT SCREEN
if st.session_state.phase == "consent":
    st.markdown("""
    <style>
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

    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    body {
        background: #ffffff !important;
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    [data-testid="stMarkdown"] p,
    [data-testid="stMarkdown"] li,
    [data-testid="stMarkdown"] span,
    [data-testid="stMarkdown"] div,
    [data-testid="stMarkdown"] h1,
    [data-testid="stMarkdown"] h2,
    [data-testid="stMarkdown"] h3,
    [data-testid="stMarkdown"] h4,
    [data-testid="stMarkdown"] h5,
    [data-testid="stMarkdown"] h6 {
        font-size: 1.125rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .stApp div[data-testid="stButton"] button[data-testid="stBaseButton-primary"],
    .stApp div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {
        margin-bottom: 1.25rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Research Consent")
    st.markdown("**Please read the consent information below carefully. Participation is voluntary.**")

    with st.expander("Key Information", expanded=True):
        st.markdown("""
            - You are being invited to participate in a research study. Participation is completely voluntary.
            - Purpose: to examine how adults infer and interpret cause and effect and how adults understand the thoughts and feelings of other people.
            - Sessions: 1–4 testing sessions (usually one), each ≤ 30 minutes. You will play interactive games; sometimes you may receive payment incentives based on your choices. You may view short clips, images, or music and answer related questions.
            - Risks: primarily the risk of breach of confidentiality.
            - Benefits: no direct benefits. Results may improve our understanding of causal reasoning and social cognition.
        """)

    st.markdown("### Purpose of the Study")
    st.markdown("""
        This study is conducted by Alison Gopnik (UC Berkeley) and research staff.
        It investigates how adults infer and interpret cause and effect and how adults understand the thoughts and feelings of other people.
        Participation is entirely voluntary.
    """)

    st.markdown("### Study Procedures")
    st.markdown("""
        Up to 4 testing sessions (usually one), each ≤ 30 minutes. Tasks may include causal learning, linguistic, imagination, categorization/association, and general cognitive tasks. You may be asked to make judgments, answer questions, observe events, and perform actions (e.g., grouping objects or activating machines). You can skip any question and withdraw at any time without penalty. Attention checks ensure data quality; failure may result in rejection and no compensation.
    """)

    st.markdown("### Benefits")
    st.markdown("While there is no direct benefit, you may enjoy the interactive displays and contribute to science.")

    st.markdown("### Risks/Discomforts")
    st.markdown("We do not expect foreseeable risks beyond a minimal risk of confidentiality breach; safeguards are in place to minimize this risk.")

    st.markdown("### Confidentiality")
    st.markdown("""
        Your identity will be separated from your data and a random code used for tracking. Data are kept indefinitely and stored securely (encrypted, restricted access). Identifiers may be removed for future research use without additional consent. Your personal information may be released if required by law. Authorized representatives (e.g., sponsors such as NSF, Mind Science Foundation, Princeton University, Defense Advanced Research) may review data for study oversight.
    """)

    st.markdown("### Costs of Study Participation")
    st.markdown("There are no costs associated with study participation.")

    st.markdown("### Compensation")
    st.markdown("""
        For your participation in our research, you will receive a maximum rate of 8 per hour. Payment ranges from 0.54 to 0.67 for a 5-minute task and from 3.25 to 4.00 for a 30-minute task, depending on the time it takes to complete the type of task you've been assigned. For studies on Prolific, you will receive a minimum rate of 6.50 per hour. For experiments with a differential bonus payment system you may have the opportunity to earn "points" that are worth up to 5 cents each, with a total bonus of no more than 30 cents paid on top of the flat fee paid for the task completion. Your online account will be credited directly.
    """)

    st.markdown("### Rights")
    st.markdown("""
        Participation is voluntary. You are free to withdraw your consent and discontinue participation at any time without penalty or loss of benefits to which you are otherwise entitled.
    """)

    st.markdown("### Questions")
    st.markdown("""
        If you have any questions, please contact the lab at gopniklab@berkeley.edu or the project lead, Eunice Yiu, at ey242@berkeley.edu.
        If you have questions regarding your treatment or rights as a participant, contact the Committee for the Protection of Human Subjects at UC Berkeley at (510) 642-7461 or subjects@berkeley.edu.
        If you have questions about the software, please contact Mandana Samiei, at mandana.samiei@mail.mcgill.ca.
    """)

    if IRB_PROTOCOL_NUMBER:
        st.markdown(f"**IRB Protocol Number:** {IRB_PROTOCOL_NUMBER}")

    st.markdown(
        '> By selecting the "Accept" button below, I acknowledge that I am 18 or older, have read this consent form, and I agree to take part in the research. If you do NOT agree, click "Decline".'
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

elif st.session_state.phase == "no_consent":
    st.title("Thanks for your response")
    st.markdown("## You did not consent. The study will now close.")
    st.stop()

# 1) PARTICIPANT ID ENTRY SCREEN
if st.session_state.phase == "intro":
    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    body {
        background: #ffffff !important;
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    [data-testid="stMarkdown"] p,
    [data-testid="stMarkdown"] li,
    [data-testid="stMarkdown"] span,
    [data-testid="stMarkdown"] div,
    [data-testid="stMarkdown"] h1,
    [data-testid="stMarkdown"] h2,
    [data-testid="stMarkdown"] h3,
    [data-testid="stMarkdown"] h4,
    [data-testid="stMarkdown"] h5,
    [data-testid="stMarkdown"] h6 {
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Welcome to the Experiment")

    if firebase_initialized:
        print("✅ Firebase connected - Data saving enabled")
    else:
        print("⚠️ Firebase not connected - Running in demo mode")

    if not st.session_state.participant_id_entered:
        st.markdown("""
**Welcome!**

In this experiment, you’ll see a machine that can light up when certain objects are placed on it.
Your task is to figure out **which object or combination of objects** makes the machine turn on.

You’ll complete several rounds:
1.  **A brief comprehension check** to make sure you understand the controls.
2.  **A short practice round** with a simple rule.
3.  **Three main rounds**, each with a new machine and new objects.

Click **“Continue”** to begin.

""")
        participant_id = st.text_input("Prolific ID:", key="participant_id")
        col_a, col_b = st.columns(2)
        with col_a:
            age_options = list(range(18, 100))
            age = st.selectbox("Age:", age_options, index=0, key="participant_age")
        with col_b:
            gender = st.selectbox(
                "Gender:",
                ["Prefer not to say", "Female", "Male", "Non-binary", "Other"],
                index=0,
                key="participant_gender",
            )

        if st.button("Continue", type="primary"):
            if not participant_id.strip():
                st.warning("Please enter your Prolific ID to continue.")
                st.stop()

            st.session_state.current_participant_id = participant_id.strip()
            st.session_state.participant_id_entered = True

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
                        'age': int(st.session_state.get('participant_age', 25)),
                        'gender': st.session_state.get('participant_gender', 'Prefer not to say'),
                    })
                except Exception:
                    pass

            st.session_state.phase = "comprehension"
            st.rerun()

    st.stop()

# 2) COMPREHENSION PHASE
elif st.session_state.phase == "comprehension":
    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        width: clamp(360px, 42vw, 780px) !important;
        max-width: clamp(360px, 42vw, 780px) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
        padding: 1.5rem !important;
    }
    body {
        background: #ffffff !important;
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    [data-testid="stMarkdown"] p,
    [data-testid="stMarkdown"] li,
    [data-testid="stMarkdown"] span,
    [data-testid="stMarkdown"] div,
    [data-testid="stMarkdown"] h1,
    [data-testid="stMarkdown"] h2,
    [data-testid="stMarkdown"] h3,
    [data-testid="stMarkdown"] h4,
    [data-testid="stMarkdown"] h5,
    [data-testid="stMarkdown"] h6 {
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧙 Blicket Text Adventure")

    if not st.session_state.comprehension_completed:
        st.markdown("## 🧠 Comprehension Phase")
        st.markdown(f"**Hello {st.session_state.current_participant_id}!**")
        st.markdown("""
        This phase helps you learn the interface.

        **Instructions:**
        - You will see 3 objects. Click to place them on the machine.
        - Select one or more objects, then click "Test".
        - If the machine lights up, that combination works.
        - Your tests and outcomes appear in the Test History (left sidebar).

        Click below when ready.
        """)

        if st.button("Start Comprehension Phase", type="primary"):
            random_blicket = random.randint(0, 2)
            st.session_state.practice_blicket_index = random_blicket
            practice_config = {
                'num_objects': 3,
                'num_blickets': 1,
                'blicket_indices': [random_blicket],
                'rule': 'conjunctive',
                'init_prob': 0.2,
                'transition_noise': 0.0,
                'horizon': 5
            }

            env, first_obs = create_new_game(
                seed=999,
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

            save_intermediate_progress_app(
                st.session_state.current_participant_id,
                "comprehension_game_start",
                1, 1, 0
            )

            st.rerun()

    st.stop()

# 3) PRACTICE GAME
elif st.session_state.phase == "practice_game":
    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        width: clamp(360px, 42vw, 780px) !important;
        max-width: clamp(360px, 42vw, 780px) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
        padding: 1.5rem !important;
    }
    body {
        background: #ffffff !important;
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    [data-testid="stMarkdown"] p,
    [data-testid="stMarkdown"] li,
    [data-testid="stMarkdown"] span,
    [data-testid="stMarkdown"] div,
    [data-testid="stMarkdown"] h1,
    [data-testid="stMarkdown"] h2,
    [data-testid="stMarkdown"] h3,
    [data-testid="stMarkdown"] h4,
    [data-testid="stMarkdown"] h5,
    [data-testid="stMarkdown"] h6 {
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧙 Blicket Text Adventure")
    st.markdown("## Comprehension Phase - Round 1")

    practice_blicket = st.session_state.get("practice_blicket_index")
    if practice_blicket is None:
        practice_blicket = random.randint(0, 2)
        st.session_state.practice_blicket_index = practice_blicket
    practice_config = {
        'num_objects': 3,
        'num_blickets': 1,
        'blicket_indices': [practice_blicket],
        'rule': 'conjunctive',
        'init_prob': 0.2,
        'transition_noise': 0.0,
        'horizon': 5
    }

    def comprehension_save_func(participant_id, game_data):
        game_data['phase'] = 'comprehension'
        game_data['interface_type'] = st.session_state.interface_type
        save_game_data(participant_id, game_data)

    textual_blicket_game_page(
        st.session_state.current_participant_id,
        practice_config,
        0,
        1,
        comprehension_save_func,
        use_visual_mode=False,
        is_practice=True
    )

# 4) PRACTICE COMPLETE
elif st.session_state.phase == "practice_complete":
    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        width: clamp(360px, 42vw, 780px) !important;
        max-width: clamp(360px, 42vw, 780px) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
        padding: 1.5rem !important;
    }
    body {
        background: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧙 Blicket Text Adventure")
    st.markdown("## Comprehension Phase Complete!")
    st.markdown(f"**Great job, {st.session_state.current_participant_id}!**")
    
    # Reveal the true blicket from the practice round
    practice_blicket = st.session_state.get("practice_blicket_index")
    if practice_blicket is not None:
        st.markdown(f"The true blicket in the practice round was **Object {practice_blicket + 1}**.")

    if st.button("Start Main Experiment", type="primary", use_container_width=True):
        random.seed(hash(st.session_state.current_participant_id) % 2**32)

        num_rounds = 3
        round_configs = []
        current_rule = random.choice(['conjunctive', 'disjunctive'])
        rule_change_probability = 0.4

        blicket_combinations = [
            [0, 1], [1, 2], [0, 2], [2, 3], [0, 3], [1, 3],
        ]
        random.shuffle(blicket_combinations)

        for i in range(num_rounds):
            num_objects = 4
            blicket_indices = blicket_combinations[i % len(blicket_combinations)]
            num_blickets = len(blicket_indices)

            if i == 0:
                rule = current_rule
            else:
                if random.random() < rule_change_probability:
                    available_rules = ['conjunctive', 'disjunctive']
                    available_rules.remove(current_rule)
                    rule = random.choice(available_rules)
                    current_rule = rule
                else:
                    rule = current_rule

            init_prob = random.uniform(0.1, 0.3)
            transition_noise = 0.0

            round_configs.append({
                'num_objects': num_objects,
                'num_blickets': num_blickets,
                'blicket_indices': blicket_indices,
                'rule': rule,
                'init_prob': init_prob,
                'transition_noise': transition_noise,
                'horizon': 16
            })

        # Ensure there is at least one conjunctive and one disjunctive round
        rules_present = {cfg['rule'] for cfg in round_configs}
        if len(rules_present) == 1:
            # Flip the rule of the final round to introduce variety
            alternate_rule = 'disjunctive' if round_configs[-1]['rule'] == 'conjunctive' else 'conjunctive'
            round_configs[-1]['rule'] = alternate_rule

        config = {
            'num_rounds': num_rounds,
            'user_selected_objects': 4,
            'rounds': round_configs,
            'interface_type': 'text',
            'demographics': {
                'prolific_id': st.session_state.get('current_participant_id', ''),
                'age': int(st.session_state.get('participant_age', 25)),
                'gender': st.session_state.get('participant_gender', 'Prefer not to say'),
            }
        }
        save_participant_config(st.session_state.current_participant_id, config)

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

        save_intermediate_progress_app(
            st.session_state.current_participant_id,
            "comprehension",
            0, 1, len(st.session_state.get('user_actions', []))
        )

        st.session_state.steps_taken = 0
        st.session_state.user_actions = []
        st.session_state.action_history = []
        st.session_state.state_history = []
        st.session_state.selected_objects = set()

        st.session_state.pop("visual_game_state", None)
        st.session_state.pop("rule_hypothesis", None)
        st.session_state.pop("rule_type", None)
        for i in range(10):
            st.session_state.pop(f"blicket_q_{i}", None)

        save_intermediate_progress_app(
            st.session_state.current_participant_id,
            "main_experiment_start",
            1, num_rounds, 0
        )

        st.session_state.phase = "game"
        st.rerun()

# 5) MAIN GAME
elif st.session_state.phase == "game":
    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        width: clamp(360px, 46vw, 860px) !important;
        max-width: clamp(360px, 46vw, 860px) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
        padding: 1.5rem !important;
    }
    body {
        background: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧙 Blicket Text Adventure")

    round_config = st.session_state.round_configs[st.session_state.current_round]

    def save_visual_game_data(participant_id, game_data):
        game_data['phase'] = 'main_experiment'
        game_data['interface_type'] = st.session_state.interface_type
        save_game_data(participant_id, game_data)

    textual_blicket_game_page(
        st.session_state.current_participant_id,
        round_config,
        st.session_state.current_round,
        len(st.session_state.round_configs),
        save_visual_game_data,
        use_visual_mode=False
    )

# NEXT ROUND
elif st.session_state.phase == "next_round":
    st.session_state.current_round += 1

    st.session_state.steps_taken = 0
    st.session_state.user_actions = []
    st.session_state.action_history = []
    st.session_state.state_history = []
    st.session_state.selected_objects = set()

    st.session_state.round_hypothesis = ""
    st.session_state.round_rule_type = ""

    st.session_state.pop("rule_hypothesis", None)
    st.session_state.pop("rule_type", None)

    st.session_state.phase = "game"
    st.rerun()

# LEGACY QA
elif st.session_state.phase == "qa":
    st.markdown(f"## 📝 Round {st.session_state.current_round + 1} Q&A")
    for i, question in enumerate(BINARY_QUESTIONS):
        st.radio(question, ("Yes", "No"), key=f"qa_{i}")

    if st.session_state.current_round + 1 < len(st.session_state.round_configs):
        st.button("Submit & Continue to Next Round", on_click=submit_qa)
    else:
        st.button("Submit & Finish", on_click=submit_qa)

# END SCREEN
elif st.session_state.phase == "end":
    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        width: clamp(360px, 42vw, 780px) !important;
        max-width: clamp(360px, 42vw, 780px) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
        padding: 1.5rem !important;
    }
    body {
        background: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("## 🎉 All done!")
    st.markdown(f"Thanks for playing, {st.session_state.current_participant_id}!")
    st.markdown("""
    ### 🎯 Experiment Complete!

    Thank you for participating in our blicket research study!
    """)
    st.button("Start Over", on_click=reset_all)

    # Add right sidebar HTML element - only on desktop
    import streamlit.components.v1 as components
    components.html("""
    <div id="right-sidebar" style="
        position: fixed;
        top: 0;
        right: 0;
        width: 320px;
        height: 100vh;
        background-color: #f0f2f6;
        border-left: 1px solid #e0e0e0;
        z-index: 1000;
        display: block;
    "></div>
    <script>
        function checkScreenSize() {
            if (window.innerWidth <= 768) {
                document.getElementById('right-sidebar').style.display = 'none';
            } else {
                document.getElementById('right-sidebar').style.display = 'block';
            }
        }
        window.addEventListener('resize', checkScreenSize);
        checkScreenSize(); // Check on load
    </script>
    """, height=0)
