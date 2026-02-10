"""
Survey app: collect human data without comprehension or exploration.
- No comprehension phase.
- Action history is provided as text only (upload or paste).
- Participants answer blicket questions and rule inference only; no object interaction.
- All data saved to Firebase under participant_id/survey.
"""
import os
import re
import datetime
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Nexiom Survey", layout="wide")

# Guard print against BrokenPipeError in Streamlit teardown
import builtins as _builtins

def _safe_print(*args, **kwargs):
    try:
        _builtins.print(*args, **kwargs)
    except (BrokenPipeError, Exception):
        pass

print = _safe_print

IRB_PROTOCOL_NUMBER = os.getenv("IRB_PROTOCOL_NUMBER", "")
COMPLETION_CODE = os.getenv("SURVEY_COMPLETION_CODE", "C1C28QBX")
PROLIFIC_RETURN_URL = f"https://app.prolific.com/submissions/complete?cc={COMPLETION_CODE}"

DEFAULT_NUM_OBJECTS = 4

# ————— Firebase —————
firebase_initialized = False
db_ref = None

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
            database_url = st.secrets["firebase"]["database_url"]
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
            database_url = None
            firebase_credentials = None

        if database_url and firebase_credentials and firebase_credentials.get("project_id"):
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            db_ref = db.reference()
            firebase_initialized = True
            print("Firebase initialized for survey app")
    except Exception as e:
        print(f"Firebase initialization failed: {e}")
        import traceback
        traceback.print_exc()
else:
    firebase_initialized = True
    db_ref = db.reference()


def save_demographics(participant_id, demographics):
    """Save participant demographics to Firebase."""
    if not firebase_initialized or not db_ref:
        return
    try:
        ref = db_ref.child(participant_id)
        ref.child("demographics").set(demographics)
        ref.child("created_at").set(datetime.datetime.now().isoformat())
        ref.child("status").set("demographics")
        ref.child("app_type").set("survey_no_exploration")
        print(f"Saved demographics for {participant_id}")
    except Exception as e:
        print(f"Failed to save demographics: {e}")


def save_consent(participant_id):
    """Save consent to Firebase."""
    if not firebase_initialized or not db_ref:
        return
    try:
        ref = db_ref.child(participant_id)
        ref.child("consent").set({
            "given": True,
            "timestamp": st.session_state.get("consent_timestamp", ""),
            "irb_protocol_number": IRB_PROTOCOL_NUMBER,
        })
        print(f"Saved consent for {participant_id}")
    except Exception as e:
        print(f"Failed to save consent: {e}")


def save_survey_data(participant_id, action_history_text, num_objects, steps, blicket_answers, rule_hypothesis, rule_type):
    """Save survey response to Firebase. Action history stored as text only."""
    if not firebase_initialized or not db_ref:
        return
    try:
        ref = db_ref.child(participant_id)
        now = datetime.datetime.now()
        payload = {
            "phase": "survey_no_exploration",
            "action_history_text": action_history_text,
            "num_objects": num_objects,
            "num_steps": len(steps),
            "blicket_classifications": blicket_answers,
            "rule_hypothesis": (rule_hypothesis or "").strip(),
            "rule_type": rule_type or "",
            "saved_at": now.isoformat(),
            "session_timestamp": now.timestamp(),
            "app_type": "survey_no_exploration",
        }
        ref.child("survey").set(payload)
        ref.child("status").set("completed")
        ref.child("completed_at").set(now.isoformat())
        print(f"Saved survey data for {participant_id}")
    except Exception as e:
        print(f"Failed to save survey data: {e}")
        import traceback
        traceback.print_exc()


def parse_action_history(content: str):
    """Parse txt content into list of steps. Each step: {'action': str, 'machine': 'ON'|'OFF'|None}.
    Supports:
    - 'action N: <description>,' with optional '-> Nexiom machine is ON/OFF'
    - Legacy: '# num_objects=4' and 'action | Machine: ON/OFF'
    """
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    num_objects = DEFAULT_NUM_OBJECTS
    for raw in content.splitlines():
        raw = raw.strip()
        if raw.startswith("#") and "num_objects=" in raw:
            m = re.search(r"num_objects\s*=\s*(\d+)", raw, re.IGNORECASE)
            if m:
                num_objects = max(1, min(10, int(m.group(1))))
            break
    steps = []
    max_object_seen = 0
    for line in lines:
        action_part = line
        machine = None
        # Format: "action N: ... -> Nexiom machine is ON/OFF," or "action N: ...,"
        m_action = re.match(r"action\s+\d+\s*:\s*(.+)", line, re.IGNORECASE)
        if m_action:
            action_part = m_action.group(1).rstrip(",").strip()
            # Check for "-> Nexiom machine is ON" or "Nexiom machine is OFF"
            machine_match = re.search(r"->\s*Nexiom machine is (ON|OFF)", action_part, re.IGNORECASE)
            if machine_match:
                machine = "ON" if machine_match.group(1).upper() == "ON" else "OFF"
            # Infer object count from "Object N" in text
            for obj_m in re.finditer(r"Object\s+(\d+)", action_part, re.IGNORECASE):
                max_object_seen = max(max_object_seen, int(obj_m.group(1)))
        else:
            # Legacy: "action | Machine: ON/OFF"
            if "|" in line:
                parts = line.split("|", 1)
                action_part = parts[0].strip()
                rest = parts[1].strip()
                if re.match(r"machine\s*:\s*(on|off)", rest, re.IGNORECASE):
                    machine = "ON" if rest.lower().split(":")[-1].strip().startswith("on") else "OFF"
        steps.append({"action": action_part, "machine": machine})
    if max_object_seen > 0:
        num_objects = max(num_objects, max_object_seen, 1)
        num_objects = min(num_objects, 10)
    return steps, num_objects


def render_history(steps):
    """Render action history: only the action text; show machine ON/OFF only when we have it."""
    if not steps:
        st.info("No steps. Add action history above (upload .txt or paste text).")
        return
    st.markdown(
        "<div style='text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 12px; padding: 10px; background-color: #555; color: #fff; border-radius: 6px;'>Action history</div>",
        unsafe_allow_html=True,
    )
    for i, step in enumerate(steps):
        machine = step.get("machine")
        machine_line = ""
        if machine is not None:
            machine_color = "#388e3c" if machine == "ON" else "#333"
            machine_line = f"<div style='font-size: 15px; font-weight: bold; color: {machine_color}'>Machine: {machine}</div>"
        st.markdown(
            f"""
            <div style='width: 100%; margin: 8px 0; padding: 12px 16px; background-color: #e8e8e8; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box;'>
                <div style='font-size: 16px; font-weight: bold; margin-bottom: 6px;'>Step {i + 1}</div>
                <div style='font-size: 15px;'>{step["action"]}</div>
                {machine_line}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ————— Session state —————
if "phase" not in st.session_state:
    st.session_state.phase = "consent"
if "consent" not in st.session_state:
    st.session_state.consent = None
if "consent_timestamp" not in st.session_state:
    st.session_state.consent_timestamp = None
if "current_participant_id" not in st.session_state:
    st.session_state.current_participant_id = ""
if "participant_id_entered" not in st.session_state:
    st.session_state.participant_id_entered = False
if "survey_submitted" not in st.session_state:
    st.session_state.survey_submitted = False

# ————— Global CSS: center content like main app —————
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
    background: #ffffff !important;
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
    background-color: #ffffff !important;
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
</style>
""", unsafe_allow_html=True)

# ————— Consent (same as main app app.py) —————
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

if st.session_state.phase == "no_consent":
    st.title("Thanks for your response")
    st.markdown("## You did not consent. The study will now close.")
    st.stop()

# ————— Intro (same as main app app.py) —————
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
        print("Firebase connected - Data saving enabled")
    else:
        print("Firebase not connected - Running in demo mode")

    if not st.session_state.participant_id_entered:
        st.markdown("""
        **Welcome!**

        In this experiment, you'll see a series of actions that were used by an active explorer to test a "Nexiom" machine to see which object (or combination of objects) makes it turn **ON**. 
        After reading the action history, you will be asked to answer questions related to the objects and the machine. 

        Click **"Continue"** to begin.

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
                save_consent(st.session_state.current_participant_id)
                save_demographics(st.session_state.current_participant_id, {
                    "prolific_id": st.session_state.current_participant_id,
                    "age": int(st.session_state.get("participant_age", 25)),
                    "gender": str(st.session_state.get("participant_gender", "Prefer not to say")),
                })

            st.session_state.phase = "action_history"
            st.rerun()

    st.stop()

# ————— Action history + Q&A —————
if st.session_state.phase == "action_history":
    st.title("Nexiom: Action history and questions")
    st.markdown("Provide the action history (upload or paste), then answer the object and rule questions. No object interaction or exploration.")

    st.header("1. Action history")
    input_mode = st.radio(
        "How do you want to provide the action history?",
        ["Upload a .txt file", "Paste text"],
        horizontal=True,
        key="input_mode",
    )
    content = None
    if input_mode == "Upload a .txt file":
        uploaded = st.file_uploader("Choose a .txt file", type=["txt"], key="upload")
        if uploaded:
            content = uploaded.read().decode("utf-8", errors="replace")
    else:
        content = st.text_area(
            "Paste your action history (one step per line, e.g. 'action 1: Placed Object 2 on machine,' or 'action 2: Test the machine -> Nexiom machine is ON,')",
            height=120,
            placeholder="action 1: Placed Object 2 on machine,\naction 2: Test the machine -> Nexiom machine is ON,\naction 3: Removed Object 2 from machine,",
            key="paste_history",
        )

    if content:
        steps, num_objects = parse_action_history(content)
        st.session_state["survey_steps"] = steps
        st.session_state["survey_num_objects"] = num_objects
        st.session_state["survey_action_history_text"] = content
    else:
        steps = st.session_state.get("survey_steps", [])
        num_objects = st.session_state.get("survey_num_objects", DEFAULT_NUM_OBJECTS)

    if not steps:
        st.info("Upload a .txt file or paste action history above to continue.")
        st.stop()

    st.header("2. Action history on screen")
    render_history(steps)

    st.header("3. Object identification and rule inference")
    st.markdown("Based on the action history above, answer the following. No exploration or object interaction.")

    st.subheader("Object identification")
    st.markdown("For each object, indicate whether you think it is a **Nexiom** (can make the machine turn on).")
    blicket_answers = {}
    for i in range(num_objects):
        blicket_answers[f"object_{i}"] = st.radio(
            f"Is Object {i + 1} a Nexiom?",
            ["Yes", "No"],
            key=f"survey_blicket_q_{i}",
            index=None,
        )

    st.subheader("Rule inference")
    rule_hypothesis = st.text_area(
        "Describe how you think the objects turn on the Nexiom machine.",
        placeholder="e.g. Both object 1 and object 2 need to be on the machine.",
        height=100,
        key="survey_rule_hypothesis",
    )
    rule_type = st.radio(
        "What type of rule do you think applies?",
        ["Conjunctive", "Disjunctive"],
        key="survey_rule_type",
        index=None,
    )

    all_answered = all(blicket_answers.get(f"object_{i}") is not None for i in range(num_objects))
    can_submit = all_answered and rule_type is not None

    if st.button("Submit responses", type="primary", disabled=not can_submit, use_container_width=True):
        action_history_text = st.session_state.get("survey_action_history_text", "")
        save_survey_data(
            st.session_state.current_participant_id,
            action_history_text,
            num_objects,
            steps,
            blicket_answers,
            (rule_hypothesis or "").strip(),
            rule_type or "",
        )
        st.session_state.survey_submitted = True
        st.session_state.phase = "end"
        st.rerun()
    st.stop()

# ————— End —————
if st.session_state.phase == "end":
    st.title("Survey complete")
    st.markdown(f"Thanks for participating, {st.session_state.current_participant_id}!")
    st.markdown(f"""
    ### Experiment complete

    Your completion code is **{COMPLETION_CODE}**.
    [Click here to return to Prolific]({PROLIFIC_RETURN_URL}) to confirm completion.
    """)
    st.stop()
