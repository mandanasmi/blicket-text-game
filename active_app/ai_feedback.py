"""Optional "Ask the AI for feedback" panel for the main-game exploration phase.

Enabled by NEXIOM_AI_FEEDBACK=1 (env) or a NEXIOM_AI_FEEDBACK key in the deployment's
Streamlit secrets. Off by default, so existing links are unchanged.

The AI sees only what the participant sees: the task description, the number of
objects, tests used/remaining, and the participant's own test history. It is never
given the true Nexioms or the rule, so its feedback is about the exploration itself.

Every exchange is kept in st.session_state.ai_feedback_log (copied into the main-round
record) and also written right away to Firebase under <pid>/ai_feedback/<n>.

Uses the OpenAI Responses API. API key: st.secrets["openai"]["api_key"], else the
OPENAI_API_KEY env var. Optional settings (env or secrets): NEXIOM_AI_FEEDBACK_MODEL
(default gpt-5.5), NEXIOM_AI_FEEDBACK_EFFORT (reasoning effort, default low),
NEXIOM_AI_FEEDBACK_MAX (answered requests per round, default 10; 0 = unlimited).
"""
import datetime
import os
import time

import streamlit as st

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_EFFORT = "low"
DEFAULT_MAX_REQUESTS = 10

SYSTEM_PROMPT = """You are an AI assistant helping a participant in an online study. The participant is exploring a "Nexiom machine" and asks you for feedback on how their exploration is going.

The task, as the participant was told it:
- There are several numbered objects. Some of them are Nexioms; the others are not.
- The participant places any combination of objects on the machine and presses Test. The machine then turns ON or OFF.
- Whether the machine turns on depends on which Nexioms are on it, according to a rule the participant must work out. The rule may be different from the practice round.
- The participant has a limited number of tests. Afterwards they must say which objects are Nexioms and describe the rule.

You do not know which objects are Nexioms or what the rule is. You only see the participant's test history, the same information they have.

When asked for feedback:
- Comment on what their tests so far do and do not tell them, and on how informative their strategy is.
- If useful, suggest one to three specific next tests and say what each would help distinguish.
- If they ask what you conclude, you may share your tentative reading of the evidence, but say clearly how certain it is and what is still untested.
- Never claim knowledge you do not have.
- Keep replies short: at most about 120 words, plain language, no headings. Refer to objects as "Object 1", "Object 2", and so on."""


def _setting(name, default=""):
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default


def ai_feedback_enabled():
    return _setting("NEXIOM_AI_FEEDBACK").lower() in ("1", "true", "yes")


def _max_requests():
    try:
        return int(_setting("NEXIOM_AI_FEEDBACK_MAX", str(DEFAULT_MAX_REQUESTS)))
    except ValueError:
        return DEFAULT_MAX_REQUESTS


def _model():
    return _setting("NEXIOM_AI_FEEDBACK_MODEL", DEFAULT_MODEL)


def _api_key():
    try:
        if hasattr(st, "secrets") and "openai" in st.secrets:
            key = st.secrets["openai"].get("api_key")
            if key:
                return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY") or None


@st.cache_resource
def _client(api_key):
    import openai
    return openai.OpenAI(api_key=api_key, timeout=60.0, max_retries=2)


def _history_text(state_history, num_objects, horizon):
    used = len(state_history)
    lines = [
        f"There are {num_objects} objects (Object 1 to Object {num_objects}).",
        f"Tests used: {used} of {horizon} ({horizon - used} remaining).",
    ]
    if not state_history:
        lines.append("No tests have been run yet.")
    else:
        lines.append("Test history:")
        for i, state in enumerate(state_history, 1):
            on = sorted(state.get("objects_on_machine", []))
            objs = ", ".join(f"Object {o + 1}" for o in on) if on else "no objects"
            result = "ON" if state.get("machine_lit") else "OFF"
            lines.append(f"Test {i}: {objs} -> machine {result}")
    return "\n".join(lines)


def _build_messages(log, history_text, question):
    """Earlier exchanges stay as they were sent, so the AI remembers what it said."""
    messages = []
    for entry in log:
        if entry.get("response"):
            messages.append({"role": "user", "content": entry["prompt_sent"]})
            messages.append({"role": "assistant", "content": entry["response"]})
    prompt = f"{history_text}\n\nParticipant's question: {question or 'Can you give me feedback on my exploration so far?'}"
    messages.append({"role": "user", "content": prompt})
    return messages, prompt


def _ask_ai(messages):
    """Returns (reply_text, error_text, model_used)."""
    import openai

    key = _api_key()
    if not key:
        return None, "AI feedback is not configured (no API key).", None
    model = _model()
    kwargs = dict(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=messages,
        max_output_tokens=2000,
        store=False,
    )
    if model.startswith(("gpt-5", "o")):
        kwargs["reasoning"] = {"effort": _setting("NEXIOM_AI_FEEDBACK_EFFORT", DEFAULT_EFFORT)}
    try:
        response = _client(key).responses.create(**kwargs)
    except openai.RateLimitError:
        return None, "The AI is busy right now. Please try again in a moment.", model
    except openai.APIStatusError as e:
        print(f"❌ AI feedback API error {e.status_code}: {e.message}")
        return None, "The AI could not answer right now. Please try again.", model
    except openai.APIConnectionError as e:
        print(f"❌ AI feedback connection error: {e}")
        return None, "Could not reach the AI. Please try again.", model

    text = (response.output_text or "").strip()
    if not text:
        print(f"⚠️ AI feedback empty reply (status={response.status}, incomplete={response.incomplete_details})")
        return None, "The AI returned an empty answer. Please try again.", response.model
    return text, None, response.model


def _save_entry(participant_id, index, entry):
    try:
        from firebase_admin import db
        db.reference().child(participant_id).child("ai_feedback").child(str(index)).set(entry)
    except Exception as e:
        print(f"⚠️ Could not save AI feedback entry for {participant_id}: {e}")


def reset_ai_feedback_state():
    for key in ("ai_feedback_log", "ai_feedback_round"):
        st.session_state.pop(key, None)


def ai_feedback_round_fields():
    """Fields merged into the saved main-round record."""
    if not ai_feedback_enabled():
        return {}
    log = st.session_state.get("ai_feedback_log", [])
    return {
        "ai_feedback_enabled": True,
        "ai_feedback_log": log,
        "ai_feedback_requests": len(log),
    }


def render_ai_feedback_panel(participant_id, round_config, current_round):
    """Draw the panel. Call only during main-game exploration."""
    if st.session_state.get("ai_feedback_round") != current_round:
        st.session_state.ai_feedback_log = []
        st.session_state.ai_feedback_round = current_round
    log = st.session_state.ai_feedback_log

    num_objects = round_config["num_objects"]
    horizon = round_config.get("horizon", 32)
    state_history = st.session_state.get("state_history", [])
    limit = _max_requests()
    # Failed requests do not count toward the limit.
    answered = sum(1 for e in log if e.get("response"))
    left = None if limit <= 0 else max(0, limit - answered)

    st.markdown("---")
    st.markdown("<div style='font-size: 24px; font-weight: 700;'>Ask the AI for feedback</div>", unsafe_allow_html=True)
    st.caption(
        "The AI sees the same test results you do. It is not told which objects are Nexioms or how the machine works."
        + ("" if left is None else f" Requests left: {left}.")
    )

    for entry in log:
        with st.chat_message("user"):
            st.markdown(entry["question"] or "_Feedback on my exploration so far_")
        with st.chat_message("assistant"):
            if entry.get("response"):
                st.markdown(entry["response"])
            else:
                st.markdown(f"_{entry.get('error', 'No answer.')}_")

    if left == 0:
        st.info("You have used all your AI feedback requests for this game.")
        return

    with st.form(key=f"ai_feedback_form_{current_round}", clear_on_submit=True):
        question = st.text_input(
            "Your question (optional)",
            placeholder="e.g. Which test should I try next?",
            max_chars=500,
        )
        submitted = st.form_submit_button("Ask AI for feedback")

    if not submitted:
        return

    question = (question or "").strip()
    history_text = _history_text(state_history, num_objects, horizon)
    messages, prompt = _build_messages(log, history_text, question)
    asked_at = datetime.datetime.now()
    start = time.time()
    with st.spinner("The AI is thinking..."):
        reply, error, model_used = _ask_ai(messages)

    entry = {
        "request_number": len(log) + 1,
        "asked_at": asked_at.isoformat(),
        "latency_seconds": round(time.time() - start, 2),
        "tests_done_when_asked": len(state_history),
        "question": question,
        "prompt_sent": prompt,
        "response": reply or "",
        "error": error or "",
        "model": model_used or "",
    }
    log.append(entry)
    if participant_id:
        _save_entry(participant_id, len(log) - 1, entry)
    st.rerun()
