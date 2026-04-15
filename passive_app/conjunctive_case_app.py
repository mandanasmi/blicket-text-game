import datetime
import os

import firebase_admin
import streamlit as st
from dotenv import load_dotenv
from firebase_admin import credentials, db

load_dotenv()
st.set_page_config(page_title="Nexiom Conjunctive Case", layout="centered")
st.markdown("""
<style>
h1, h2, h3 {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    word-break: break-word !important;
}
[data-testid="block-container"],
.block-container {
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)

IRB_PROTOCOL_NUMBER = os.getenv("IRB_PROTOCOL_NUMBER", "")
COMPLETION_CODE = os.getenv("SURVEY_COMPLETION_CODE", "C1C28QBX")
PROLIFIC_RETURN_URL = f"https://app.prolific.com/submissions/complete?cc={COMPLETION_CODE}"

TRAINING_STEPS = [
    "Object 1 did not turn on the Nexiom Machine.",
    "Object 2 did not turn on the Nexiom Machine.",
    "Object 3 did not turn on the Nexiom Machine.",
    "Objects 1 and 2 did not turn on the Nexiom Machine.",
    "Objects 1 and 3 turned on the Nexiom Machine.",
    "Objects 2 and 3 turned on the Nexiom Machine.",
]

TEST_STEPS = [
    "Object 4 did not turn on the Nexiom Machine.",
    "Object 4 did not turn on the Nexiom Machine.",
    "Object 4 did not turn on the Nexiom Machine.",
    "Object 5 did not turn on the Nexiom Machine.",
    "Objects 4 and 6 turned on the Nexiom Machine.",
    "Objects 4 and 6 turned on the Nexiom Machine.",
]


def _valid_database_url(url):
    if not url or not isinstance(url, str):
        return False
    return "firebaseio.com" in url or "firebasedatabase.app" in url


firebase_initialized = False
db_ref = None
firebase_init_error = None

if not firebase_admin._apps:
    try:
        firebase_credentials = None
        database_url = None
        if hasattr(st, "secrets") and hasattr(st.secrets, "firebase") and "firebase" in st.secrets:
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
                "universe_domain": "googleapis.com",
            }
            database_url = st.secrets["firebase"].get("database_url") or st.secrets["firebase"].get("databaseURL")
        elif os.getenv("FIREBASE_PROJECT_ID"):
            firebase_credentials = {
                "type": "service_account",
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": (os.getenv("FIREBASE_PRIVATE_KEY") or "").replace("\\n", "\n"),
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
                "universe_domain": "googleapis.com",
            }
            database_url = os.getenv("FIREBASE_DATABASE_URL")
        else:
            firebase_init_error = "No Firebase config found. App will run in demo mode."

        if firebase_init_error is None and database_url and not _valid_database_url(database_url):
            firebase_init_error = "Invalid FIREBASE_DATABASE_URL: use Realtime Database URL, not Firebase Console URL."

        if firebase_init_error is None and database_url and firebase_credentials and firebase_credentials.get("project_id"):
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            db_ref = db.reference()
            firebase_initialized = True
    except Exception as e:
        firebase_init_error = str(e)
else:
    try:
        firebase_initialized = True
        db_ref = db.reference()
    except Exception as e:
        firebase_init_error = str(e)


def save_consent(participant_id):
    if not firebase_initialized or not db_ref:
        return
    db_ref.child(participant_id).child("consent").set(
        {
            "given": True,
            "timestamp": st.session_state.get("consent_timestamp", ""),
            "irb_protocol_number": IRB_PROTOCOL_NUMBER,
        }
    )


def save_demographics(participant_id, age, gender):
    if not firebase_initialized or not db_ref:
        return
    ref = db_ref.child(participant_id)
    ref.child("demographics").set(
        {
            "prolific_id": participant_id,
            "age": age,
            "gender": gender,
        }
    )
    ref.child("created_at").set(datetime.datetime.now().isoformat())
    ref.child("status").set("conjunctive_case_in_progress")
    ref.child("app_type").set("conjunctive_passive_ui")


def save_conjunctive_case(
    participant_id,
    answers,
    time_per_training_step_seconds,
    time_per_test_step_seconds,
    questions_response_time_seconds,
):
    if not firebase_initialized or not db_ref:
        return
    now = datetime.datetime.now()
    entered_at = st.session_state.get("task_started_at")
    response_time_seconds = None
    if entered_at is not None:
        response_time_seconds = round(now.timestamp() - entered_at, 2)

    payload = {
        "phase": "conjunctive_case",
        "training_steps": TRAINING_STEPS,
        "test_steps": TEST_STEPS,
        "questions": [
            "Is Object 4 a Nexiom?",
            "Is Object 5 a Nexiom?",
            "Is Object 6 a Nexiom?",
        ],
        "nexiom_object_answers": answers,
        "object_answers": answers,
        "time_per_training_step_seconds": time_per_training_step_seconds,
        "time_per_test_step_seconds": time_per_test_step_seconds,
        "questions_response_time_seconds": questions_response_time_seconds,
        "saved_at": now.isoformat(),
        "submitted_at": now.isoformat(),
        "session_timestamp": now.timestamp(),
        "response_time_seconds": response_time_seconds,
        "app_type": "conjunctive_passive_ui",
    }
    ref = db_ref.child(participant_id)
    ref.child("conjunctive_case_data").set(payload)
    ref.child("status").set("completed")
    ref.child("completed_at").set(now.isoformat())


for key, value in {
    "phase": "consent",
    "consent_timestamp": None,
    "participant_id": "",
    "task_started_at": None,
    "training_index": 0,
    "test_index": 0,
    "training_step_times": [],
    "test_step_times": [],
    "training_step_entered_at": None,
    "test_step_entered_at": None,
    "questions_entered_at": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


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
            - Sessions: 1-4 testing sessions (usually one), each <= 30 minutes. You will play interactive games; sometimes you may receive payment incentives based on your choices. You may view short clips, images, or music and answer related questions.
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
        Up to 4 testing sessions (usually one), each <= 30 minutes. Tasks may include causal learning, linguistic, imagination, categorization/association, and general cognitive tasks. You may be asked to make judgments, answer questions, observe events, and perform actions (e.g., grouping objects or activating machines). You can skip any question and withdraw at any time without penalty. Attention checks ensure data quality; failure may result in rejection and no compensation.
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
        if st.button("Accept", type="primary", use_container_width=True):
            st.session_state.consent_timestamp = datetime.datetime.now().isoformat()
            st.session_state.phase = "intro"
            st.rerun()
    with col2:
        if st.button("Decline", use_container_width=True):
            st.session_state.phase = "declined"
            st.rerun()
    st.stop()


if st.session_state.phase == "declined":
    st.title("Thanks for your response")
    st.markdown("## You did not consent. The study will now close.")
    st.stop()


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
    if not firebase_initialized:
        st.warning("Firebase is not connected. Responses will not be saved.")
        if firebase_init_error:
            st.code(firebase_init_error, language=None)

    st.markdown("""
    **Welcome!**

    In this experiment, you'll read a sequence of observations about objects tested on a "Nexiom" machine to see which object (or combination of objects) makes it turn ON. **Only objects that are "Nexioms" will make the machine turn on.** **If the Nexiom machine switches on, it means that the objects on the machine satisfied the activation rule.** If the machine stays off, it means the objects on the machine did not satisfy the activation rule. After reading the training and test observations, you will answer questions about whether specific objects are Nexioms.

    **Your Task:**
    You will read through a series of test results one by one. There is a training sequence and a test sequence. Please pay close attention to which objects (or combinations) were used and what the outcome was. Afterward, we will ask you to judge whether specific new objects are likely to be Nexioms.

    Click **"Start"** to begin.
    """)

    participant_id = st.text_input("Prolific ID", value=st.session_state.participant_id)
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.selectbox("Age", list(range(18, 100)), index=0)
    with col_b:
        gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Non-binary", "Other"], index=0)

    if st.button("Start", type="primary"):
        if not participant_id.strip():
            st.warning("Please enter your Prolific ID.")
            st.stop()
        st.session_state.participant_id = participant_id.strip()
        now_ts = datetime.datetime.now().timestamp()
        st.session_state.task_started_at = now_ts
        st.session_state.training_index = 0
        st.session_state.test_index = 0
        st.session_state.training_step_times = []
        st.session_state.test_step_times = []
        st.session_state.training_step_entered_at = now_ts
        st.session_state.test_step_entered_at = None
        st.session_state.questions_entered_at = None
        if firebase_initialized and db_ref:
            save_consent(st.session_state.participant_id)
            save_demographics(st.session_state.participant_id, int(age), str(gender))
        st.session_state.phase = "training"
        st.rerun()
    st.stop()


if st.session_state.phase == "training":
    idx = st.session_state.training_index
    if st.session_state.training_step_entered_at is None:
        st.session_state.training_step_entered_at = datetime.datetime.now().timestamp()
    st.title("Training")
    st.progress((idx + 1) / len(TRAINING_STEPS))
    st.markdown(f"### {TRAINING_STEPS[idx]}")
    if st.button("Next", type="primary", use_container_width=True):
        now_ts = datetime.datetime.now().timestamp()
        entered = st.session_state.training_step_entered_at
        if entered is not None:
            st.session_state.training_step_times.append(round(now_ts - entered, 2))
        if idx < len(TRAINING_STEPS) - 1:
            st.session_state.training_index += 1
            st.session_state.training_step_entered_at = now_ts
        else:
            st.session_state.phase = "test"
            st.session_state.test_step_entered_at = now_ts
            st.session_state.training_step_entered_at = None
        st.rerun()
    st.stop()


if st.session_state.phase == "test":
    idx = st.session_state.test_index
    if st.session_state.test_step_entered_at is None:
        st.session_state.test_step_entered_at = datetime.datetime.now().timestamp()
    st.title("Test")
    st.progress((idx + 1) / len(TEST_STEPS))
    st.markdown(f"### {TEST_STEPS[idx]}")
    if st.button("Next", type="primary", use_container_width=True):
        now_ts = datetime.datetime.now().timestamp()
        entered = st.session_state.test_step_entered_at
        if entered is not None:
            st.session_state.test_step_times.append(round(now_ts - entered, 2))
        if idx < len(TEST_STEPS) - 1:
            st.session_state.test_index += 1
            st.session_state.test_step_entered_at = now_ts
        else:
            st.session_state.phase = "questions"
            st.session_state.questions_entered_at = now_ts
            st.session_state.test_step_entered_at = None
        st.rerun()
    st.stop()


if st.session_state.phase == "questions":
    if st.session_state.questions_entered_at is None:
        st.session_state.questions_entered_at = datetime.datetime.now().timestamp()
    st.title("Questions")
    q4 = st.radio("Is Object 4 a Nexiom?", ["Yes", "No"], index=None, key="q_obj4")
    q5 = st.radio("Is Object 5 a Nexiom?", ["Yes", "No"], index=None, key="q_obj5")
    q6 = st.radio("Is Object 6 a Nexiom?", ["Yes", "No"], index=None, key="q_obj6")

    all_answered = all(v is not None for v in [q4, q5, q6])
    if st.button("Submit", type="primary", disabled=not all_answered, use_container_width=True):
        answers = {"object_4": q4, "object_5": q5, "object_6": q6}
        now_ts = datetime.datetime.now().timestamp()
        q_started = st.session_state.questions_entered_at
        questions_response_time_seconds = (
            round(now_ts - q_started, 2) if q_started is not None else None
        )
        if firebase_initialized and db_ref:
            save_conjunctive_case(
                st.session_state.participant_id,
                answers,
                list(st.session_state.get("training_step_times", [])),
                list(st.session_state.get("test_step_times", [])),
                questions_response_time_seconds,
            )
        st.session_state.phase = "end"
        st.rerun()
    st.stop()


if st.session_state.phase == "end":
    st.title("Experiment complete")
    st.markdown("Your responses have been recorded. Thank you for participating.")
    st.markdown("Your Prolific completion code is:")
    st.code(COMPLETION_CODE, language=None)
    st.markdown(f"[Return to Prolific]({PROLIFIC_RETURN_URL})")
