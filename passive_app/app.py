"""
Survey app: collect human data without comprehension or exploration.
- No comprehension phase.
- Action history is provided as text only (upload or paste).
- Participants answer blicket questions and rule inference only; no object interaction.
- All data saved to Firebase under participant_id/survey.
"""
import os
import re
import html
import datetime
import streamlit as st
import streamlit.components.v1 as components
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Nexiom Text Adventure", layout="wide")

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

# ————— Assigned action histories (one per use, cycle through 102 files) —————
# Path to active_explore/analysis/action_histories relative to this app file
_ACTION_HISTORIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "active_explore", "analysis", "action_histories")
_ACTION_HISTORY_FILES = None  # Lazy-loaded sorted list of filenames

def get_action_history_file_list():
    """Return sorted list of *_action_history.txt filenames (full path). Deterministic order."""
    global _ACTION_HISTORY_FILES
    if _ACTION_HISTORY_FILES is None:
        if not os.path.isdir(_ACTION_HISTORIES_DIR):
            _ACTION_HISTORY_FILES = []
        else:
            _ACTION_HISTORY_FILES = sorted(
                f for f in os.listdir(_ACTION_HISTORIES_DIR)
                if f.endswith("_action_history.txt")
            )
            _ACTION_HISTORY_FILES = [os.path.join(_ACTION_HISTORIES_DIR, f) for f in _ACTION_HISTORY_FILES]
    return _ACTION_HISTORY_FILES

def get_next_action_history_index():
    """
    Return (index, filepath, filename) for the next action history to use.
    Uses Firebase _config/action_history_next_index (transaction) to cycle 0..101.
    If Firebase is not connected, returns (0, first_file_path, first_filename) and does not persist.
    """
    files = get_action_history_file_list()
    n = len(files)
    if n == 0:
        return None, None, None
    if not firebase_initialized or not db_ref:
        # No Firebase: use first file every time (for local testing)
        return 0, files[0], os.path.basename(files[0])
    try:
        ref = db_ref.child("_config").child("action_history_next_index")

        def updater(current):
            if current is None:
                current = 0
            return current + 1

        new_value = ref.transaction(updater)
        index = (new_value - 1) % n
        filepath = files[index]
        filename = os.path.basename(filepath)
        return index, filepath, filename
    except Exception as e:
        print(f"get_next_action_history_index failed: {e}")
        return 0, files[0], os.path.basename(files[0])

# ————— Firebase —————
firebase_initialized = False
db_ref = None
firebase_init_error = None

def _valid_database_url(url):
    """Realtime Database URL must contain firebaseio.com or firebasedatabase.app, not the Console page."""
    if not url or not isinstance(url, str):
        return False
    return "firebaseio.com" in url or "firebasedatabase.app" in url

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
            database_url = None
            firebase_credentials = None
            firebase_init_error = "No Firebase config: set Streamlit secrets [firebase] or env vars (FIREBASE_PROJECT_ID, FIREBASE_DATABASE_URL, etc.). See passive_app/FIREBASE.md."

        if firebase_init_error is None and database_url and not _valid_database_url(database_url):
            firebase_init_error = (
                "database_url must be your Realtime Database URL (e.g. https://PROJECT-default-rtdb.firebaseio.com), "
                "not the Firebase Console page URL. In Firebase Console: Build -> Realtime Database, copy the URL."
            )

        if firebase_init_error is None and database_url and firebase_credentials and firebase_credentials.get("project_id"):
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            db_ref = db.reference()
            firebase_initialized = True
            print("Firebase initialized for passive app")
    except Exception as e:
        firebase_init_error = str(e)
        print(f"Firebase initialization failed: {e}")
        import traceback
        traceback.print_exc()
else:
    try:
        firebase_initialized = True
        db_ref = db.reference()
    except Exception as e:
        firebase_init_error = str(e)
        firebase_initialized = False
        db_ref = None


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


def save_game_data(
    participant_id,
    action_history_text,
    num_objects,
    steps,
    blicket_answers,
    rule_hypothesis,
    rule_type,
    response_time_seconds=None,
    action_history_review_time_seconds=None,
    passive_exploration_time_seconds=None,
    time_per_step_seconds=None,
    uploaded_filename=None,
    source_participant_id=None,
):
    """Save game/survey response to Firebase under game_data. No demographics in payload."""
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
            "object_answers": blicket_answers,
            "blicket_classifications": blicket_answers,
            "rule_inference": (rule_hypothesis or "").strip(),
            "rule_type": rule_type or "",
            "saved_at": now.isoformat(),
            "submitted_at": now.isoformat(),
            "session_timestamp": now.timestamp(),
            "app_type": "survey_no_exploration",
        }
        if response_time_seconds is not None:
            payload["response_time_seconds"] = round(response_time_seconds, 2)
        if action_history_review_time_seconds is not None:
            payload["action_history_review_time_seconds"] = round(action_history_review_time_seconds, 2)
        if passive_exploration_time_seconds is not None:
            payload["passive_exploration_time"] = round(passive_exploration_time_seconds, 2)
        if time_per_step_seconds is not None and len(time_per_step_seconds) > 0:
            payload["time_per_step_seconds"] = time_per_step_seconds
        if uploaded_filename:
            payload["uploaded_filename"] = uploaded_filename
        if source_participant_id:
            payload["source_participant_id"] = source_participant_id
        ref.child("game_data").set(payload)
        ref.child("status").set("completed")
        ref.child("completed_at").set(now.isoformat())
        print(f"Saved game_data for {participant_id}")
    except Exception as e:
        print(f"Failed to save game_data: {e}")
        import traceback
        traceback.print_exc()


def parse_action_history(content: str):
    """Parse txt content into list of steps. Each step: {'action': str, 'machine': 'ON'|'OFF'|None, 'objects_on_machine': [int]|None}.
    Supports:
    - 'action N: <description>,' with optional '-> Nexiom machine is ON/OFF'
    - Legacy: '# num_objects=4' and 'action | Machine: ON/OFF'
    For Test steps, objects_on_machine lists which objects were on the machine at test time.
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
    current_objects = set()
    for line in lines:
        action_part = line
        machine = None
        objects_on_machine = None
        # Format: "action N: ... -> Nexiom machine is ON/OFF," or "action N: ...,"
        m_action = re.match(r"action\s+\d+\s*:\s*(.+)", line, re.IGNORECASE)
        if m_action:
            action_part = m_action.group(1).rstrip(",").strip()
            raw_action = action_part
            # Extract machine ON/OFF; remove from action text (we show "Action: Machine: ON/OFF" in render with bold color)
            # Match "-> Nexiom machine is ON/OFF" or "-> Machine: ON/OFF"
            machine_match = re.search(r"->\s*(?:Nexiom machine is|Machine:)\s*(ON|OFF)", action_part, re.IGNORECASE)
            if machine_match:
                machine = "ON" if machine_match.group(1).upper() == "ON" else "OFF"
                action_part = re.sub(r"->\s*(?:Nexiom machine is|Machine:)\s*(?:ON|OFF)\s*,?\s*", "", action_part, flags=re.IGNORECASE).strip()
                objects_on_machine = sorted(current_objects)
            # Update current_objects for Place/Remove
            placed = re.search(r"Placed\s+Object\s+(\d+)\s+on\s+machine", raw_action, re.IGNORECASE)
            removed = re.search(r"Removed\s+Object\s+(\d+)\s+from\s+machine", raw_action, re.IGNORECASE)
            if placed:
                current_objects.add(int(placed.group(1)))
            elif removed:
                current_objects.discard(int(removed.group(1)))
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
                    objects_on_machine = sorted(current_objects)
        steps.append({"action": action_part, "machine": machine, "objects_on_machine": objects_on_machine})
    if max_object_seen > 0:
        num_objects = max(num_objects, max_object_seen, 1)
        num_objects = min(num_objects, 10)
    return steps, num_objects


def render_history(steps):
    """Render action history in one bar; action text and machine status (ON green, OFF black). Font size matches rest of app."""
    if not steps:
        st.info("No steps. Add action history above (upload .txt or paste text).")
        return
    st.markdown(
        "<div style='text-align: center; font-size: 1.125rem; font-weight: bold; margin-bottom: 8px; padding: 8px 12px; background-color: #555; color: #fff; border-radius: 6px;'>Action history</div>",
        unsafe_allow_html=True,
    )
    parts = []
    for i, step in enumerate(steps):
        machine = step.get("machine")
        machine_bit = ""
        if machine is not None:
            machine_color = "#388e3c" if machine == "ON" else "#000000"
            machine_bit = f": <span style='font-weight: bold; color: {machine_color}'>Machine: {machine}</span>"
        action_escaped = html.escape(step["action"])
        parts.append(
            f"<div style='font-size: 1.05rem; margin: 6px 0; line-height: 1.4;'>"
            f"<span style='font-weight: bold;'>Step {i + 1}</span> {action_escaped}{machine_bit}"
            f"</div>"
        )
    inner = "\n".join(parts)
    st.markdown(
        f"""
        <div style='width: 100%; padding: 14px 16px; background-color: #e8e8e8; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; font-size: 1.05rem;'>
            {inner}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_single_step(step, step_num):
    """Render one step (action + machine status) in the same style as render_history."""
    machine = step.get("machine")
    machine_bit = ""
    if machine is not None:
        machine_color = "#388e3c" if machine == "ON" else "#000000"
        machine_bit = f": <span style='font-weight: bold; color: {machine_color}'>Machine: {machine}</span>"
    action_escaped = html.escape(step["action"])
    st.markdown(
        f"""
        <div style='width: 100%; padding: 14px 16px; background-color: #e8e8e8; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; font-size: 1.05rem;'>
            <div style='font-size: 1.05rem; line-height: 1.4;'>
                <span style='font-weight: bold;'>Step {step_num}</span> {action_escaped}{machine_bit}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_step_html(step, step_num, is_current=False, max_chars=80, num_objects=4):
    """One line for sidebar: 'Step N: action text' with machine ON in green or OFF. Current step in blue.
    For Test steps: show a summary box matching active game style (object Yes/No + Machine result)."""
    action = (step.get("action") or "").strip()
    if len(action) > max_chars:
        action = action[: max_chars - 3].rstrip() + "..."
    action_escaped = html.escape(action)
    machine = step.get("machine")
    objects_on_machine = step.get("objects_on_machine") or set()
    if isinstance(objects_on_machine, list):
        objects_on_machine = set(objects_on_machine)
    machine_bit = ""
    test_summary = ""
    if machine is not None:
        if machine == "ON":
            machine_bit = " <span style='color: #388e3c; font-weight: bold;'>ON</span>"
            machine_color = "#388e3c"
        else:
            machine_bit = " <span style='color: #000; font-weight: bold;'>OFF</span>"
            machine_color = "#333333"
        if objects_on_machine is not None:
            objects_set = set(objects_on_machine)
            objects_text = ""
            for obj_id in range(1, num_objects + 1):
                is_on_platform = obj_id in objects_set
                yes_no = "Yes" if is_on_platform else "No"
                bg_color = "#ffffff" if is_on_platform else "#d0d0d0"
                border_style = "2px solid #999" if is_on_platform else "1px solid #ccc"
                objects_text += (
                    "<span style='display: inline-flex; align-items: center; justify-content: center; "
                    f"background-color: {bg_color}; color: black; padding: 4px 6px; margin: 2px; border-radius: 5px; "
                    f"font-size: 14px; font-weight: bold; border: {border_style}; min-width: 44px;'>"
                    f"<span style='margin-right: 3px;'>{obj_id}</span>"
                    f"<span>{yes_no}</span></span>"
                )
            test_summary = (
                f"<div style='margin: 6px 0 8px 0; padding: 8px 10px; background-color: #a9a9a9; "
                f"color: #1f1f1f; border: 1px solid #7f7f7f; border-radius: 8px; "
                f"box-shadow: 0 2px 4px rgba(0,0,0,0.15); font-size: 12px;'>"
                f"<div style='margin-bottom: 6px; display: flex; flex-wrap: wrap; justify-content: center; gap: 6px;'>{objects_text}</div>"
                f"<div style='font-weight: bold; font-size: 14px; color: {machine_color};'>Machine: {machine}</div>"
                f"</div>"
            )
    color_style = "color: #1976d2;" if is_current else ""
    return f"<div style='margin-bottom: 12px; line-height: 1.4; font-size: 1.1rem; {color_style}'><strong>Step {step_num}:</strong> {action_escaped}{machine_bit}{test_summary}</div>"


def render_test_history_sidebar(steps, current_index=None, num_objects=None):
    """Render Action History panel in sidebar. Title in a separate box; steps below (empty at start), scrollable."""
    if num_objects is None:
        num_objects = DEFAULT_NUM_OBJECTS
    with st.sidebar:
        st.markdown(
            "<div style='text-align: center; font-size: 1.125rem; font-weight: bold; margin-top: -10px; margin-bottom: 10px; padding: 10px 12px; background-color: #b0b0b0; border: 1px solid #999; border-radius: 6px; color: #1a1a1a;'>Action History</div>",
            unsafe_allow_html=True,
        )
        _box_shadow = "0 4px 14px rgba(0,0,0,0.2)"
        if current_index == -1:
            # Begin screen: empty bar, no steps shown
            st.markdown(
                f"""
                <div style="max-height: 80vh; overflow-y: auto; overflow-x: hidden; padding-right: 8px; padding-bottom: 32px; font-size: 1.1rem; box-sizing: border-box; min-height: 60px; background-color: #e0e0e0; border-radius: 6px; box-shadow: {_box_shadow};">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            parts = []
            # Only show steps that have been revealed by pressing Next (no placeholders for unseen steps)
            if current_index is None:
                # Full list (e.g. identification screen): show all steps
                end_idx = len(steps)
            else:
                end_idx = current_index + 1
            for i in range(end_idx):
                step = steps[i]
                step_num = i + 1
                is_current = current_index is not None and i == current_index
                parts.append(_sidebar_step_html(step, step_num, is_current=is_current, num_objects=num_objects))
            inner = "\n".join(parts)
            st.markdown(
                f"""
                <div id="action-history-scroll" style="max-height: 80vh; overflow-y: auto; overflow-x: hidden; padding-right: 8px; padding-bottom: 32px; font-size: 1.1rem; box-sizing: border-box; background-color: #e0e0e0; border-radius: 6px; box-shadow: {_box_shadow};">
                    {inner}
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Auto-scroll to bottom when new steps are added (so latest action is visible)
            _scroll_script = """
            <script>
            (function() {
                function scrollToBottom() {
                    var el = parent.document.getElementById("action-history-scroll");
                    if (el) { el.scrollTop = el.scrollHeight; }
                }
                if (document.readyState === "complete") scrollToBottom();
                else parent.addEventListener("load", scrollToBottom);
                setTimeout(scrollToBottom, 50);
            })();
            </script>
            """
            components.html(_scroll_script, height=0)


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
if "survey_sequence_viewed" not in st.session_state:
    st.session_state.survey_sequence_viewed = False
if "survey_action_history_entered_at" not in st.session_state:
    st.session_state.survey_action_history_entered_at = None
if "demographics" not in st.session_state:
    st.session_state.demographics = None
if "survey_first_object_response_at" not in st.session_state:
    st.session_state.survey_first_object_response_at = None
if "survey_uploaded_filename" not in st.session_state:
    st.session_state.survey_uploaded_filename = None
if "survey_source_participant_id" not in st.session_state:
    st.session_state.survey_source_participant_id = None
if "survey_assigned_file_index" not in st.session_state:
    st.session_state.survey_assigned_file_index = None
if "survey_passive_exploration_start_at" not in st.session_state:
    st.session_state.survey_passive_exploration_start_at = None
if "survey_passive_exploration_time_seconds" not in st.session_state:
    st.session_state.survey_passive_exploration_time_seconds = None
if "survey_history_step_index" not in st.session_state:
    st.session_state.survey_history_step_index = 0
if "survey_step_view_times" not in st.session_state:
    st.session_state.survey_step_view_times = []  # seconds spent on each step before clicking Next
if "survey_last_step_entered_at" not in st.session_state:
    st.session_state.survey_last_step_entered_at = None

# ————— Global CSS: center content like main app —————
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stMain"],
main, main section,
[data-testid="stVerticalBlock"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

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
    box-shadow: none !important;
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

[data-testid="stExpander"],
.stExpander,
[data-testid="stVerticalBlock"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
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
        st.warning("Firebase is not connected. Data will not be saved.")
        if firebase_init_error:
            st.code(firebase_init_error, language=None)
        st.markdown("Check Streamlit Secrets (or `.streamlit/secrets.toml`) and that Realtime Database is enabled. **database_url** must be the Realtime Database URL (e.g. `https://PROJECT-default-rtdb.firebaseio.com`), not the Firebase Console page. See passive_app/FIREBASE.md.")

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
                demographics = {
                    "prolific_id": st.session_state.current_participant_id,
                    "age": int(st.session_state.get("participant_age", 25)),
                    "gender": str(st.session_state.get("participant_gender", "Prefer not to say")),
                }
                save_demographics(st.session_state.current_participant_id, demographics)
                st.session_state.demographics = demographics

            st.session_state.phase = "action_history"
            st.rerun()

    st.stop()

# ————— Action history + Q&A —————
if st.session_state.phase == "action_history":
    # Same background as consent; slightly wider sidebar for Action History bar
    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
    }
    body { background: #ffffff !important; }
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 380px !important;
        max-width: 380px !important;
    }
    section.main .stButton > button {
        background-color: #1976d2 !important;
        border-color: #1976d2 !important;
        color: white !important;
    }
    section.main .stButton > button:hover {
        background-color: #1565c0 !important;
        border-color: #1565c0 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("🧙 Nexiom Text Adventure")

    # Assign one action history file per use (cycle through 102 files via Firebase counter)
    if st.session_state.survey_assigned_file_index is None:
        index, filepath, filename = get_next_action_history_index()
        if filepath and os.path.isfile(filepath):
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                st.error(f"Could not load assigned action history: {e}")
                content = ""
            if content:
                steps, num_objects = parse_action_history(content)
                st.session_state["survey_steps"] = steps
                st.session_state["survey_num_objects"] = num_objects
                st.session_state["survey_action_history_text"] = content
                st.session_state["survey_history_step_index"] = -1
                st.session_state["survey_step_view_times"] = []
                st.session_state["survey_last_step_entered_at"] = None
                st.session_state["survey_uploaded_filename"] = filename
                if "_action_history" in filename:
                    st.session_state["survey_source_participant_id"] = filename.split("_action_history")[0].strip()
                else:
                    st.session_state["survey_source_participant_id"] = filename.replace(".txt", "").strip() or None
                st.session_state["survey_assigned_file_index"] = index
        else:
            st.session_state["survey_assigned_file_index"] = -1  # No files; don't retry
            st.info("No action history files found. Add files to active_explore/analysis/action_histories/ or run locally with that directory.")
            st.stop()

    steps = st.session_state.get("survey_steps", [])
    num_objects = st.session_state.get("survey_num_objects", DEFAULT_NUM_OBJECTS)
    content = st.session_state.get("survey_action_history_text", "")

    if not steps:
        st.info("No action history loaded.")
        st.stop()

    # Time from entering this section until they press "Next: Object identification"
    if st.session_state.survey_passive_exploration_start_at is None:
        st.session_state.survey_passive_exploration_start_at = datetime.datetime.now().timestamp()

    # Progressive reveal: begin screen, then step-by-step, then full history screen, then identification questions
    if not st.session_state.get("survey_sequence_viewed", False):
        step_index = st.session_state.get("survey_history_step_index", -1)
        if step_index < len(steps):
            # Begin screen (step_index == -1) or step-by-step view
            if step_index >= 0 and st.session_state.survey_last_step_entered_at is None:
                st.session_state.survey_last_step_entered_at = datetime.datetime.now().timestamp()
            render_test_history_sidebar(steps, step_index if step_index >= 0 else -1, num_objects=num_objects)
            st.header("Action history")
            if step_index < 0:
                # Initial screen: no action shown yet
                st.markdown(
                    "You will see the action history one by one. Press **Next** to see each action. "
                    "Once you observe an action, it will appear in the action history bar on the left for future review if necessary."
                )
            else:
                st.markdown(
                    "You can access the action history one by one by pressing the **Next** button. "
                    "Once you observe an action, it will appear in the action history bar on the left for future review if necessary."
                )
                st.markdown(f"**Step {step_index + 1} of {len(steps)}**")
                render_single_step(steps[step_index], step_index + 1)
            st.markdown("<div id='step-next-spacer' style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<style>#step-next-spacer ~ div .stButton > button { background-color: #1976d2 !important; border-color: #1976d2 !important; }</style>",
                unsafe_allow_html=True,
            )
            next_clicked = st.button("Next", type="primary", use_container_width=True)
            if next_clicked:
                if step_index >= 0 and st.session_state.survey_last_step_entered_at is not None:
                    duration = datetime.datetime.now().timestamp() - st.session_state.survey_last_step_entered_at
                    st.session_state.survey_step_view_times.append(round(duration, 2))
                st.session_state.survey_last_step_entered_at = None
                if step_index < len(steps) - 1:
                    st.session_state.survey_history_step_index = step_index + 1
                else:
                    st.session_state.survey_history_step_index = len(steps)
                st.rerun()
            st.stop()
        else:
            # After last step: no "Action history" in main area; message and button to identification
            render_test_history_sidebar(steps, current_index=None, num_objects=num_objects)
            st.markdown(
                "You have seen all steps. There are no more actions to view. "
                "Please proceed to answer the questions below."
            )
            if st.button("Next: Object identification", type="primary", use_container_width=True):
                st.session_state.survey_sequence_viewed = True
                if st.session_state.survey_action_history_entered_at is None:
                    st.session_state.survey_action_history_entered_at = datetime.datetime.now().timestamp()
                start_at = st.session_state.get("survey_passive_exploration_start_at")
                if start_at is not None:
                    st.session_state.survey_passive_exploration_time_seconds = (
                        datetime.datetime.now().timestamp() - start_at
                    )
                st.rerun()
            st.stop()

    # survey_sequence_viewed: left bar has history; main area is questions only
    render_test_history_sidebar(steps, current_index=None, num_objects=num_objects)

    st.header("3. Object identification")
    st.markdown("Based on the action history in the left bar, indicate for each object whether you think it is a **Nexiom** (can make the machine turn on).")

    blicket_answers = {}
    for i in range(num_objects):
        blicket_answers[f"object_{i}"] = st.radio(
            f"Is Object {i + 1} a Nexiom?",
            ["Yes", "No"],
            key=f"survey_blicket_q_{i}",
            index=None,
        )

    # Time spent reviewing action history before first object answer
    if st.session_state.survey_first_object_response_at is None and blicket_answers.get("object_0") is not None:
        st.session_state.survey_first_object_response_at = datetime.datetime.now().timestamp()

    st.header("4. Rule inference")
    def _rerun_on_rule_change():
        st.rerun()

    rule_hypothesis = st.text_area(
        "Describe how you think the objects turn on the Nexiom machine.",
        height=100,
        key="survey_rule_hypothesis",
        on_change=_rerun_on_rule_change,
    )

    all_answered = all(blicket_answers.get(f"object_{i}") is not None for i in range(num_objects))
    # Enable "Next: Rule type" as soon as at least one character is written (any character)
    rule_inference_filled = len(rule_hypothesis or "") >= 1

    if st.button("Next: Rule type", type="primary", disabled=not (all_answered and rule_inference_filled), use_container_width=True):
        st.session_state.saved_rule_hypothesis = (rule_hypothesis or "").strip()
        st.session_state.phase = "rule_inference"
        st.rerun()
    st.stop()

# ————— Rule type selection (new page) —————
if st.session_state.phase == "rule_inference":
    st.markdown("""
    <style>
    [data-testid="block-container"],
    .block-container {
        background-color: transparent !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        min-height: auto !important;
    }
    body { background: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)
    st.title("Rule type")
    st.markdown("Based on the action history and your answers, what type of rule do you think governs this Nexiom machine?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Conjunctive rule**

        The machine switches on when **ALL** of the Nexioms are present on the machine.

        *Example: If Objects 1 and 3 are Nexioms, the machine only switches on when BOTH Object 1 AND Object 3 are on the machine.*
        """)
    with col2:
        st.markdown("""
        **Disjunctive rule**

        The machine switches on when **ANY** of the Nexioms are present on the machine.

        *Example: If Objects 1 and 3 are Nexioms, the machine switches on when EITHER Object 1 OR Object 3 (or both) are on the machine.*
        """)

    rule_type_raw = st.radio(
        "What type of rule do you think applies?",
        ["Conjunctive (ALL Nexioms must be present)", "Disjunctive (ANY Nexiom can activate)"],
        key="survey_rule_type",
        index=None,
    )
    rule_type = "Conjunctive" if rule_type_raw and "Conjunctive" in rule_type_raw else ("Disjunctive" if rule_type_raw and "Disjunctive" in rule_type_raw else None)

    if st.button("Submit responses", type="primary", disabled=rule_type is None, use_container_width=True):
        steps = st.session_state.get("survey_steps", [])
        num_objects = st.session_state.get("survey_num_objects", DEFAULT_NUM_OBJECTS)
        blicket_answers = {
            f"object_{i}": st.session_state.get(f"survey_blicket_q_{i}", "No")
            for i in range(num_objects)
        }
        action_history_text = st.session_state.get("survey_action_history_text", "")
        rule_hypothesis = st.session_state.get("saved_rule_hypothesis", "")
        entered_at = st.session_state.get("survey_action_history_entered_at")
        first_object_at = st.session_state.get("survey_first_object_response_at")
        response_time_seconds = (datetime.datetime.now().timestamp() - entered_at) if entered_at else None
        action_history_review_time_seconds = (
            (first_object_at - entered_at) if (entered_at and first_object_at) else None
        )
        save_game_data(
            st.session_state.current_participant_id,
            action_history_text,
            num_objects,
            steps,
            blicket_answers,
            rule_hypothesis,
            rule_type or "",
            response_time_seconds=response_time_seconds,
            action_history_review_time_seconds=action_history_review_time_seconds,
            passive_exploration_time_seconds=st.session_state.get("survey_passive_exploration_time_seconds"),
            time_per_step_seconds=st.session_state.get("survey_step_view_times", []),
            uploaded_filename=st.session_state.get("survey_uploaded_filename"),
            source_participant_id=st.session_state.get("survey_source_participant_id"),
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
