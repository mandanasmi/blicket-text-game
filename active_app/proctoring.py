"""Full-screen gate, focus tracking and bonus instructions for proctored deployments.

Enabled per deployment with NEXIOM_PROCTORED=1 (set by the act-noc/act-sid wrappers).
A browser page cannot block screenshots or a second device, so this is a deterrent plus
measurement: participants must stay in full screen, copy/paste and right-click are
blocked, and every full-screen exit, tab switch, window blur, blocked copy/paste and
PrintScreen press is logged per phase and saved to Firebase under <pid>/integrity.
"""
import datetime
import os

import streamlit as st
import streamlit.components.v1 as components

_component = components.declare_component(
    "nexiom_proctor",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "proctor_component"),
)

# Phases where the full-screen gate and tracking are switched off.
_INACTIVE_PHASES = {"consent", "no_consent", "integrity", "end"}


def proctoring_enabled():
    return os.getenv("NEXIOM_PROCTORED", "").strip().lower() in ("1", "true", "yes")


def render_integrity_page():
    """Instructions page shown right after consent. Returns True once accepted."""
    st.title("Before you begin")
    st.markdown(
        '<ol style="font-size:1.15rem; line-height:1.7;">'
        "<li>Please stay in <b>full screen</b> and do not switch to other tabs throughout the "
        "experiment, otherwise <u>you risk losing your compensation</u>.</li>"
        "<li>You will receive a <b>bonus payment</b> if you correctly identify how the machine works.</li>"
        "</ol>",
        unsafe_allow_html=True,
    )
    agreed = st.checkbox("I understand.")
    if st.button("Continue", type="primary", disabled=not agreed):
        st.session_state.integrity_accepted_at = datetime.datetime.now().isoformat()
        return True
    return False


def grade_bonus(true_blicket_indices, user_chosen_blickets, true_rule, rule_type_answer):
    """Bonus = every Nexiom classified correctly AND the right conjunctive/disjunctive rule."""
    nexioms_correct = sorted(int(i) for i in true_blicket_indices) == sorted(int(i) for i in user_chosen_blickets)
    rule_correct = bool(rule_type_answer) and rule_type_answer.strip().lower().startswith(str(true_rule).lower())
    return {
        "bonus_nexioms_correct": nexioms_correct,
        "bonus_rule_type_correct": rule_correct,
        "bonus_eligible": nexioms_correct and rule_correct,
    }


def render_proctor(phase_label, participant_id, db_ref=None):
    """Mount the proctoring component and persist any new integrity stats.

    Call once per script run, before the phase is rendered. Stats live in the browser
    page, so they survive Streamlit reruns; Python keeps the latest snapshot in
    st.session_state.integrity_stats and mirrors it to Firebase <pid>/integrity.
    """
    if not proctoring_enabled():
        return
    stats = st.session_state.get("integrity_stats") or {}
    known = stats.get("version", 0)
    phase = phase_label.split(":", 1)[0]
    try:
        value = _component(
            phase=phase_label,
            active=phase not in _INACTIVE_PHASES,
            known_version=known,
            key="nexiom_proctor",
            default=None,
        )
    except Exception as e:
        # Never let proctoring take the study down (e.g. pyarrow missing on Cloud).
        print(f"❌ Proctoring component failed: {e}")
        return
    if isinstance(value, dict) and value.get("version", 0) > known:
        st.session_state.integrity_stats = value
        stats = value
    if (
        participant_id
        and db_ref is not None
        and stats
        and stats.get("version", 0) > st.session_state.get("integrity_saved_version", -1)
    ):
        try:
            payload = dict(stats)
            payload["accepted_at"] = st.session_state.get("integrity_accepted_at")
            db_ref.child(participant_id).child("integrity").set(payload)
            st.session_state.integrity_saved_version = stats.get("version", 0)
        except Exception as e:
            print(f"❌ Failed to save integrity stats: {e}")


def integrity_summary():
    """Compact copy of the integrity totals to store alongside the main-round data."""
    if not proctoring_enabled():
        return {}
    stats = st.session_state.get("integrity_stats") or {}
    return {
        "integrity": {
            "counts": stats.get("counts", {}),
            "away_seconds": stats.get("away_seconds", 0),
            "by_phase": stats.get("by_phase", {}),
            "fullscreen_supported": stats.get("fullscreen_supported"),
        }
    }
