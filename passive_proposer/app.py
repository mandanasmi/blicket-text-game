"""
Passive Proposer app.

Design goal:
- Active-like interface: participants click object buttons and then "Test Machine".
- Passive-like outcomes: each test returns the matched active participant's outcome
  for that test index (not a simulation from proposed objects).
- Participants can propose exactly as many tests as their matched active participant.
"""
import csv
import datetime
import os
import re
from typing import Dict, List, Optional

import firebase_admin
import streamlit as st
from dotenv import load_dotenv
from firebase_admin import credentials, db

load_dotenv()
st.set_page_config(page_title="Passive Proposer", layout="wide")

IRB_PROTOCOL_NUMBER = os.getenv("IRB_PROTOCOL_NUMBER", "")
COMPLETION_CODE = os.getenv("SURVEY_COMPLETION_CODE", "C1C28QBX")
PROLIFIC_RETURN_URL = f"https://app.prolific.com/submissions/complete?cc={COMPLETION_CODE}"

DEFAULT_NUM_OBJECTS = 4

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_ACTIVE_ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "active_explore", "analysis")
_TEST_HISTORIES_DIR = os.path.join(_ACTIVE_ANALYSIS_DIR, "test_histories")
_ACTION_HISTORIES_DIR = os.path.join(_ACTIVE_ANALYSIS_DIR, "all_actions")
_IDS_CSV = os.path.join(_ACTION_HISTORIES_DIR, "ids.csv")

firebase_initialized = False
db_ref = None
firebase_init_error = None


def _valid_database_url(url: Optional[str]) -> bool:
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

        if database_url and not _valid_database_url(database_url):
            firebase_init_error = (
                "database_url must be a Realtime Database URL, not a Firebase Console URL."
            )

        if firebase_init_error is None and database_url and firebase_credentials and firebase_credentials.get("project_id"):
            cred = credentials.Certificate(firebase_credentials)
            firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            db_ref = db.reference()
            firebase_initialized = True
    except Exception as e:
        firebase_init_error = str(e)
else:
    try:
        db_ref = db.reference()
        firebase_initialized = True
    except Exception as e:
        firebase_init_error = str(e)


def load_active_ids() -> List[str]:
    if not os.path.isfile(_IDS_CSV):
        return []
    ids = []
    with open(_IDS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("id") or "").strip()
            if pid:
                ids.append(pid)
    return ids


def parse_test_history_line(line: str) -> Optional[Dict]:
    raw = line.strip()
    if not raw:
        return None
    m = re.match(
        r"test\s*(\d+)\s*->\s*(.+?)\s*->\s*machine\s*(on|off)\s*$",
        raw,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    step = int(m.group(1))
    active_objects_text = m.group(2).strip()
    machine = "ON" if m.group(3).lower() == "on" else "OFF"
    return {
        "test_index": step,
        "active_objects_text": active_objects_text,
        "machine_outcome": machine,
        "raw_line": raw,
    }


def load_assigned_test_sequence(active_id: str) -> List[Dict]:
    path = os.path.join(_TEST_HISTORIES_DIR, f"{active_id}_test_history.txt")
    if not os.path.isfile(path):
        return []
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            parsed = parse_test_history_line(line)
            if parsed is not None:
                rows.append(parsed)
    return rows


def _clean_action_line(raw: str) -> str:
    text = raw.strip()
    if text.endswith(","):
        text = text[:-1].rstrip()
    m = re.match(r"action\s*\d+\s*:\s*(.*)$", text, flags=re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    return text


def load_action_chunks(active_id: str) -> List[List[str]]:
    """Partition action_history into per-test chunks.

    Each chunk is the list of cleaned action lines from just after the previous
    test up to and including the current test line. Trailing actions after the
    last test (if any) are discarded.
    """
    path = os.path.join(_ACTION_HISTORIES_DIR, f"{active_id}_action_history.txt")
    if not os.path.isfile(path):
        return []
    chunks: List[List[str]] = []
    current: List[str] = []
    with open(path, "r") as f:
        for line in f:
            cleaned = _clean_action_line(line)
            if not cleaned:
                continue
            current.append(cleaned)
            if re.search(r"Test the machine", cleaned, flags=re.IGNORECASE):
                chunks.append(current)
                current = []
    return chunks


def get_next_assignment() -> Dict:
    active_ids = load_active_ids()
    valid_pairs = []
    for aid in active_ids:
        seq = load_assigned_test_sequence(aid)
        chunks = load_action_chunks(aid)
        if len(seq) > 0 and len(chunks) >= len(seq):
            valid_pairs.append((aid, seq, chunks[: len(seq)]))
    if not valid_pairs:
        return {"active_id": None, "sequence": [], "action_chunks": []}

    n = len(valid_pairs)
    if not firebase_initialized or not db_ref:
        aid, seq, chunks = valid_pairs[0]
        return {"active_id": aid, "sequence": seq, "action_chunks": chunks, "assigned_index": 0}

    try:
        ref = db_ref.child("_config").child("passive_proposer_next_index")

        def updater(current):
            if current is None:
                current = 0
            return int(current) + 1

        new_value = ref.transaction(updater)
        idx = (new_value - 1) % n
        aid, seq, chunks = valid_pairs[idx]
        return {"active_id": aid, "sequence": seq, "action_chunks": chunks, "assigned_index": idx}
    except Exception:
        aid, seq, chunks = valid_pairs[0]
        return {"active_id": aid, "sequence": seq, "action_chunks": chunks, "assigned_index": 0}


def save_demographics(participant_id: str, age: int, gender: str) -> None:
    if not firebase_initialized or not db_ref:
        return
    now = datetime.datetime.now().isoformat()
    ref = db_ref.child(participant_id)
    ref.child("demographics").set(
        {"prolific_id": participant_id, "age": age, "gender": gender}
    )
    ref.child("created_at").set(now)
    ref.child("status").set("passive_proposer_in_progress")
    ref.child("app_type").set("passive_proposer")


def save_final_payload(participant_id: str) -> None:
    if not firebase_initialized or not db_ref:
        return
    now = datetime.datetime.now()
    started_at = st.session_state.get("task_started_at")
    total_response_time = None
    if started_at is not None:
        total_response_time = round(now.timestamp() - started_at, 2)

    payload = {
        "phase": "passive_proposer",
        "app_type": "passive_proposer",
        "source_active_participant_id": st.session_state.get("matched_active_id"),
        "assigned_test_count": st.session_state.get("max_tests"),
        "proposed_tests": st.session_state.get("proposed_tests", []),
        "matched_outcomes": st.session_state.get("outcomes_seen", []),
        "object_answers": st.session_state.get("object_answers", {}),
        "nexiom_object_answers": st.session_state.get("object_answers", {}),
        "rule_inference": (st.session_state.get("rule_hypothesis") or "").strip(),
        "rule_type": st.session_state.get("rule_type") or "",
        "saved_at": now.isoformat(),
        "submitted_at": now.isoformat(),
        "session_timestamp": now.timestamp(),
    }
    if total_response_time is not None:
        payload["response_time_seconds"] = total_response_time

    ref = db_ref.child(participant_id)
    ref.child("passive_proposer_data").set(payload)
    ref.child("status").set("completed")
    ref.child("completed_at").set(now.isoformat())


def ensure_session_state() -> None:
    defaults = {
        "phase": "consent",
        "participant_id": "",
        "consent": None,
        "consent_timestamp": None,
        "matched_active_id": None,
        "active_test_sequence": [],
        "active_action_chunks": [],
        "max_tests": 0,
        "selected_objects": set(),
        "proposed_tests": [],
        "outcomes_seen": [],
        "task_started_at": None,
        "previous_test_time": None,
        "object_answers": {},
        "rule_hypothesis": "",
        "rule_hypothesis_submitted": False,
        "rule_type": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_object_controls(disabled: bool) -> None:
    st.subheader("Available Objects")
    cols = st.columns(DEFAULT_NUM_OBJECTS)
    for i in range(DEFAULT_NUM_OBJECTS):
        with cols[i]:
            object_id = i + 1
            is_selected = object_id in st.session_state.selected_objects
            label = f"Object {object_id}"
            help_text = (
                "Interaction disabled."
                if disabled
                else f"Click to {'remove' if is_selected else 'place'} this object."
            )
            if st.button(
                label,
                key=f"obj_{object_id}",
                disabled=disabled,
                help=help_text,
                use_container_width=True,
            ):
                if is_selected:
                    st.session_state.selected_objects.remove(object_id)
                else:
                    st.session_state.selected_objects.add(object_id)
                st.rerun()
            status_color = "#0F9D58" if is_selected else "#888888"
            status_text = "ON MACHINE" if is_selected else "OFF MACHINE"
            st.markdown(
                f"<div style='text-align:center;font-weight:700;"
                f"color:{status_color};font-size:0.85rem;margin-top:0.25rem;'>"
                f"{status_text}</div>",
                unsafe_allow_html=True,
            )


def append_test_and_outcome() -> None:
    now = datetime.datetime.now()
    prev = st.session_state.previous_test_time
    t_since_prev = None
    if prev is not None:
        t_since_prev = round((now - prev).total_seconds(), 2)
    st.session_state.previous_test_time = now

    selection = sorted(st.session_state.selected_objects)
    test_idx = len(st.session_state.proposed_tests) + 1
    st.session_state.proposed_tests.append(
        {
            "test_index": test_idx,
            "proposed_objects": selection,
            "proposed_objects_text": ", ".join(f"Object {x}" for x in selection),
            "timestamp": now.isoformat(),
            "time_since_previous_test_seconds": t_since_prev,
        }
    )

    matched = st.session_state.active_test_sequence[test_idx - 1]
    chunks = st.session_state.active_action_chunks or []
    matched_chunk = chunks[test_idx - 1] if test_idx - 1 < len(chunks) else []
    st.session_state.outcomes_seen.append(
        {
            "test_index": test_idx,
            "matched_active_outcome": matched["machine_outcome"],
            "matched_active_objects_text": matched["active_objects_text"],
            "matched_raw_line": matched["raw_line"],
            "matched_action_chunk": matched_chunk,
        }
    )


def render_test_panel() -> None:
    tests_done = len(st.session_state.proposed_tests)
    tests_left = st.session_state.max_tests - tests_done
    st.markdown(
        f"**Matched active participant:** `{st.session_state.matched_active_id}`  \n"
        f"**Tests proposed:** `{tests_done}/{st.session_state.max_tests}`"
    )

    current_sel = sorted(st.session_state.selected_objects)
    if current_sel:
        st.info("Current proposal: " + ", ".join(f"Object {x}" for x in current_sel))
    else:
        st.info("Current proposal: none selected")

    can_test = tests_left > 0 and len(current_sel) > 0
    if st.button("Test Machine", type="primary", disabled=not can_test, use_container_width=True):
        append_test_and_outcome()
        st.rerun()

    if tests_left > 0:
        st.caption(f"You can propose {tests_left} more test(s).")
    else:
        st.success("You completed all matched tests. Please answer the questions below.")

    if st.session_state.outcomes_seen and tests_left > 0:
        row = st.session_state.outcomes_seen[-1]
        idx = row["test_index"]
        outcome_color = "#0F9D58" if row["matched_active_outcome"] == "ON" else "#333333"
        proposed_text = st.session_state.proposed_tests[idx - 1]["proposed_objects_text"]
        st.markdown(f"### Round {idx} — active explorer's actions")
        st.markdown(
            f"Your proposal: `{proposed_text}`",
            unsafe_allow_html=True,
        )
        chunk = row.get("matched_action_chunk") or []
        if chunk:
            lines = [f"{i}. {action}" for i, action in enumerate(chunk, start=1)]
            st.markdown("\n".join(lines))
        st.markdown(
            f"Outcome: <span style='color:{outcome_color};font-weight:700'>"
            f"Machine {row['matched_active_outcome']}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Earlier rounds are kept in the History panel on the left.")


def render_questionnaire() -> None:
    if len(st.session_state.proposed_tests) < st.session_state.max_tests:
        return

    st.markdown("---")
    st.subheader("Final Questions")
    for i in range(1, DEFAULT_NUM_OBJECTS + 1):
        key = f"answer_obj_{i}"
        val = st.radio(
            f"Is Object {i} a Nexiom?",
            options=["Yes", "No"],
            index=None,
            key=key,
            horizontal=True,
        )
        if val is not None:
            st.session_state.object_answers[f"object_{i}"] = val

    hypothesis_value = st.text_area(
        "In one sentence, describe your current rule hypothesis.",
        value=st.session_state.rule_hypothesis,
        disabled=st.session_state.rule_hypothesis_submitted,
    )
    if not st.session_state.rule_hypothesis_submitted:
        st.session_state.rule_hypothesis = hypothesis_value
        submit_hypothesis_disabled = not hypothesis_value.strip()
        if st.button(
            "Submit hypothesis",
            key="submit_hypothesis_btn",
            disabled=submit_hypothesis_disabled,
        ):
            st.session_state.rule_hypothesis = hypothesis_value.strip()
            st.session_state.rule_hypothesis_submitted = True
            st.rerun()
        st.caption("Submit your hypothesis to continue.")   
        return

    st.session_state.rule_type = st.radio(
        "Which rule type best matches your hypothesis?",
        options=["Conjunctive", "Disjunctive", "Other / Unsure"],
        index=None if not st.session_state.rule_type else ["Conjunctive", "Disjunctive", "Other / Unsure"].index(st.session_state.rule_type),
        key="rule_type_radio",
        horizontal=True,
    ) or ""

    all_answered = len(st.session_state.object_answers) == DEFAULT_NUM_OBJECTS
    rule_filled = bool(st.session_state.rule_hypothesis.strip())
    type_filled = bool(st.session_state.rule_type)
    submit_disabled = not (all_answered and rule_filled and type_filled)

    if st.button("Submit", type="primary", disabled=submit_disabled):
        save_final_payload(st.session_state.participant_id)
        st.session_state.phase = "complete"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if submit_disabled:
        st.caption("Complete all object answers and rule type to submit.")


def render_consent_page() -> None:
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
    .stApp div[data-testid="stButton"] button[data-testid="stBaseButton-primary"],
    .stApp div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {
        margin-bottom: 1.25rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("Research Consent")
    st.markdown("**Please read the consent information below carefully. Participation is voluntary.**")

    with st.expander("Key Information", expanded=True):
        st.markdown(
            """
            - You are being invited to participate in a research study. Participation is completely voluntary.
            - Purpose: to examine how adults infer and interpret cause and effect and how adults understand the thoughts and feelings of other people.
            - Sessions: 1-4 testing sessions (usually one), each <= 30 minutes. You will play interactive games; sometimes you may receive payment incentives based on your choices. You may view short clips, images, or music and answer related questions.
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
        Up to 4 testing sessions (usually one), each <= 30 minutes. Tasks may include causal learning, linguistic, imagination, categorization/association, and general cognitive tasks. You may be asked to make judgments, answer questions, observe events, and perform actions (e.g., grouping objects or activating machines). You can skip any question and withdraw at any time without penalty. Attention checks ensure data quality; failure may result in rejection and no compensation.
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
        For your participation in our research, you will receive a maximum rate of 8 per hour. Payment ranges from 0.54 to 0.67 for a 5-minute task and from 3.25 to 4.00 for a 30-minute task, depending on the time it takes to complete the type of task you've been assigned. For studies on Prolific, you will receive a minimum rate of 6.50 per hour. For experiments with a differential bonus payment system you may have the opportunity to earn "points" that are worth up to 5 cents each, with a total bonus of no more than 30 cents paid on top of the flat fee paid for the task completion. Your online account will be credited directly.
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
        If you have any questions, please contact the lab at gopniklab@berkeley.edu or the project lead, Eunice Yiu, at ey242@berkeley.edu.
        If you have questions regarding your treatment or rights as a participant, contact the Committee for the Protection of Human Subjects at UC Berkeley at (510) 642-7461 or subjects@berkeley.edu.
        If you have questions about the software, please contact Mandana Samiei, at mandana.samiei@mail.mcgill.ca.
    """
    )

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
            st.session_state.phase = "demographics"
            st.rerun()
    with col2:
        if st.button("Decline", type="secondary"):
            st.session_state.consent = False
            st.session_state.consent_timestamp = datetime.datetime.now().isoformat()
            st.session_state.phase = "no_consent"
            st.rerun()


def render_no_consent_page() -> None:
    st.title("Thanks for your response")
    st.markdown("## You did not consent. The study will now close.")


def render_consent_page_old() -> None:
    # Deprecated. Kept only as historical placeholder.
    st.markdown(
        """
You will propose tests on a Nexiom machine interface.
For each test you propose, the outcome shown is taken from a matched active participant's test at the same index.

You can propose exactly as many tests as your matched participant performed.
After that, you will answer object and rule questions.
"""
    )
    if st.checkbox("I consent to participate."):
        if st.button("Continue", type="primary"):
            st.session_state.phase = "demographics"
            st.rerun()


def render_demographics_page() -> None:
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

    st.markdown("""
**Welcome!**

In this experiment, you'll see a series of actions that were used by an active explorer to test a "Nexiom" machine to see which object (or combination of objects) makes it turn ON. **Only objects that are "Nexioms" will make the machine turn on.** **If the Nexiom machine switches on, it means that at least one of the objects you put on the machine is a Nexiom.** It could be just one of them, some of them, or all of them.

In this task, you will act as an advisor for someone who is trying to figure out which objects are Nexioms, which are objects that make the machine turn on.

On each round, you will suggest a test by choosing which objects should be placed on the machine. Your goal is to propose tests that are most useful for figuring out which objects are Nexioms.

After you submit your suggestion, you will see the test that your advisee actually carried out, along with the outcome (whether the machine turned on or off).

The advisee may or may not follow your suggestion exactly. Even if they choose a different test than the one you proposed, you should continue to give the best possible advice based on everything you have observed so far.

You will be evaluated based on how informative and helpful your suggested tests are for identifying the Nexioms.

Click **"Continue"** to begin.
    """)

    participant_id = st.text_input("Prolific ID:", value=st.session_state.participant_id).strip()
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.selectbox("Age:", list(range(18, 100)), index=0)
    with col_b:
        gender = st.selectbox(
            "Gender:",
            ["Prefer not to say", "Female", "Male", "Non-binary", "Other"],
            index=0,
        )

    if st.button("Continue", type="primary", disabled=not participant_id):
        assignment = get_next_assignment()
        if not assignment.get("sequence"):
            st.error("No active test histories available for assignment.")
            return
        st.session_state.participant_id = participant_id
        st.session_state.matched_active_id = assignment["active_id"]
        st.session_state.active_test_sequence = assignment["sequence"]
        st.session_state.active_action_chunks = assignment.get("action_chunks", [])
        st.session_state.max_tests = len(assignment["sequence"])
        st.session_state.phase = "task"
        st.session_state.task_started_at = datetime.datetime.now().timestamp()
        save_demographics(participant_id, age, gender)
        st.rerun()


def render_history_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<h3 style='text-align:center;margin-top:-0.75rem;margin-bottom:0.75rem;'>"
            "History</h3>",
            unsafe_allow_html=True,
        )
        if not st.session_state.outcomes_seen:
            st.caption("No rounds completed yet.")
            return
        entries_html = ""
        for row in st.session_state.outcomes_seen:
            idx = row["test_index"]
            proposed_text = st.session_state.proposed_tests[idx - 1][
                "proposed_objects_text"
            ]
            active_text = row.get("matched_active_objects_text") or "—"
            outcome = row["matched_active_outcome"]
            color = "#0F9D58" if outcome == "ON" else "#666666"
            entries_html += (
                f"<div style='"
                f"background:#ffffff;"
                f"border:1px solid #c9ccd1;"
                f"border-radius:8px;"
                f"padding:0.75rem 0.85rem;"
                f"margin-bottom:0.75rem;"
                f"box-shadow:0 1px 2px rgba(0,0,0,0.05);"
                f"'>"
                f"<div style='font-weight:700;margin-bottom:0.35rem;'>Round {idx}</div>"
                f"<div style='color:#7b2cbf;margin-bottom:0.2rem;'>You: "
                f"<code style='color:#7b2cbf;background:transparent;'>{proposed_text}</code></div>"
                f"<div style='margin-bottom:0.2rem;'>Active: "
                f"<code style='background:transparent;'>{active_text}</code></div>"
                f"<div>Outcome: <span style='color:{color};font-weight:700'>"
                f"Machine {outcome}</span></div>"
                f"</div>"
            )
        st.markdown(
            f"""
            <div style="
                max-height: 85vh;
                min-height: 55vh;
                overflow-y: auto;
                padding-right: 6px;
                background: transparent;
                font-size: 1rem;
            ">{entries_html}</div>
            """,
            unsafe_allow_html=True,
        )


def render_task_page() -> None:
    render_history_sidebar()
    st.title("Passive Proposer Task")
    with st.expander("Instructions", expanded=False):
        st.markdown(
            """
1. Select one or more objects and click **Test Machine** to submit your proposal.
2. You will then see the sequence of actions the matched active explorer took for that round, along with the test they ran and its outcome.
3. Continue until you reach your test limit, then answer the final questions.
"""
        )
    render_object_controls(
        disabled=len(st.session_state.proposed_tests) >= st.session_state.max_tests
    )
    render_test_panel()
    render_questionnaire()


def render_complete_page() -> None:
    st.title("Survey complete")
    participant_id = st.session_state.get("participant_id", "")
    if participant_id:
        st.markdown(f"Thanks for participating, {participant_id}!")
    st.markdown(
        """
### Experiment complete

Your completion code for Prolific is:
"""
    )
    st.code(COMPLETION_CODE, language=None)
    st.markdown(
        f"[Click here to return to Prolific]({PROLIFIC_RETURN_URL}) to confirm completion."
    )


_GLOBAL_CSS = """
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

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] section {
    background: #e8eaed !important;
    background-color: #e8eaed !important;
}

[data-testid="stSidebar"] {
    width: 340px !important;
    min-width: 340px !important;
    max-width: 340px !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: 340px !important;
    min-width: 340px !important;
    max-width: 340px !important;
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

.stApp div[data-testid="stButton"] button,
.stApp div[data-testid="stButton"] button[kind="primary"],
.stApp div[data-testid="stButton"] button[kind="secondary"],
.stApp div[data-testid="stButton"] button[data-testid="stBaseButton-primary"],
.stApp div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {
    background-color: #1976d2 !important;
    border-color: #1976d2 !important;
    color: #ffffff !important;
}
.stApp div[data-testid="stButton"] button:hover,
.stApp div[data-testid="stButton"] button[kind="primary"]:hover,
.stApp div[data-testid="stButton"] button[kind="secondary"]:hover,
.stApp div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover,
.stApp div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #125aa0 !important;
    border-color: #125aa0 !important;
    color: #ffffff !important;
}
.stApp div[data-testid="stButton"] button:disabled,
.stApp div[data-testid="stButton"] button[disabled] {
    background-color: #9ec5e8 !important;
    border-color: #9ec5e8 !important;
    color: #ffffff !important;
}
</style>
"""


def main() -> None:
    ensure_session_state()
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    if not firebase_initialized:
        st.warning("Firebase is not connected. Data will not be saved.")
        if firebase_init_error:
            st.code(firebase_init_error, language=None)

    phase = st.session_state.phase
    if phase == "consent":
        render_consent_page()
    elif phase == "no_consent":
        render_no_consent_page()
    elif phase == "demographics":
        render_demographics_page()
    elif phase == "task":
        render_task_page()
    else:
        render_complete_page()


if __name__ == "__main__":
    main()
