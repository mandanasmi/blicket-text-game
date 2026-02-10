"""
Streamlit app: view action history from a txt file and answer object identification
and rule inference. No object interaction; participants update the txt file externally
and load it here. Runs on a separate port from the main app (e.g. 8502).
"""
import os
import re
import json
import datetime
import streamlit as st

st.set_page_config(page_title="Nexiom – Action history & Q&A", layout="wide")

# —————  Txt file format —————
# Optional first line: # num_objects=4
# One step per line:  ACTION | Machine: ON   or   ACTION | Machine: OFF
# Or action-only lines:  place Object 1 on the machine  (machine state shown as ?)
# Blank lines and # lines are ignored.

DEFAULT_NUM_OBJECTS = 4
HISTORY_DIR = "data_txt_history"
os.makedirs(HISTORY_DIR, exist_ok=True)


def parse_action_history(content: str):
    """Parse txt content into list of steps. Each step: {'action': str, 'machine': 'ON'|'OFF'|None}."""
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    num_objects = DEFAULT_NUM_OBJECTS
    # Check for # num_objects=N in the raw content
    for raw in content.splitlines():
        raw = raw.strip()
        if raw.startswith("#") and "num_objects=" in raw:
            m = re.search(r"num_objects\s*=\s*(\d+)", raw, re.IGNORECASE)
            if m:
                num_objects = max(1, min(10, int(m.group(1))))
            break

    steps = []
    for line in lines:
        action_part = line
        machine = None
        if "|" in line:
            parts = line.split("|", 1)
            action_part = parts[0].strip()
            rest = parts[1].strip()
            if re.match(r"machine\s*:\s*(on|off)", rest, re.IGNORECASE):
                machine = "ON" if rest.lower().split(":")[-1].strip().startswith("on") else "OFF"
        steps.append({"action": action_part, "machine": machine})
    return steps, num_objects


def render_history(steps, num_objects):
    """Render action history in a Test History style block."""
    if not steps:
        st.info("No steps yet. Add lines to your action history file (e.g. \"place Object 1 on the machine | Machine: ON\") and load again.")
        return
    st.markdown(
        "<div style='text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 12px; padding: 10px; background-color: #555555; color: #ffffff; border-radius: 6px;'>Action history</div>",
        unsafe_allow_html=True,
    )
    for i, step in enumerate(steps):
        machine_label = step["machine"] if step["machine"] else "?"
        machine_color = "#388e3c" if step["machine"] == "ON" else "#333333" if step["machine"] == "OFF" else "#666"
        st.markdown(
            f"""
            <div style='
                width: 100%;
                margin: 8px 0;
                padding: 12px 16px;
                background-color: #e8e8e8;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-sizing: border-box;
            '>
                <div style='font-size: 16px; font-weight: bold; margin-bottom: 6px;'>Step {i + 1}</div>
                <div style='margin-bottom: 6px; font-size: 15px;'>{step["action"]}</div>
                <div style='font-size: 15px; font-weight: bold; color: {machine_color}'>Machine: {machine_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.title("Nexiom: Action history and questions")
    st.markdown(
        "This page shows an **action history** loaded from a text file. "
        "You do not interact with objects here; update your action history in the file, then load it below. "
        "After reviewing the history, answer the object identification and rule inference questions. "
        "**There is no comprehension or practice phase in this stage.**"
    )

    # —————  Load history —————
    st.header("1. Load action history")
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
            "Paste your action history (one step per line; optional: action | Machine: ON/OFF)",
            height=120,
            placeholder="place Object 1 on the machine | Machine: ON\ntake Object 2 off the machine | Machine: OFF",
            key="paste_history",
        )

    if content:
        steps, num_objects = parse_action_history(content)
        st.session_state["txt_steps"] = steps
        st.session_state["txt_num_objects"] = num_objects
        st.session_state["txt_content"] = content
    else:
        steps = st.session_state.get("txt_steps", [])
        num_objects = st.session_state.get("txt_num_objects", DEFAULT_NUM_OBJECTS)

    if not steps:
        st.stop()

    # —————  Show history —————
    st.header("2. Action history on screen")
    if steps:
        render_history(steps, num_objects)
    else:
        st.info("Load or paste action history above.")
        st.stop()

    # —————  Q&A —————
    st.header("3. Object identification and rule inference")
    st.markdown("Based on the action history above, answer the following.")

    num_objects = st.session_state.get("txt_num_objects", DEFAULT_NUM_OBJECTS)

    # Object identification
    st.subheader("Object identification")
    st.markdown("For each object, indicate whether you think it is a **Nexiom** (can make the machine turn on).")
    blicket_answers = {}
    for i in range(num_objects):
        blicket_answers[f"object_{i}"] = st.radio(
            f"Is Object {i + 1} a Nexiom?",
            ["Yes", "No"],
            key=f"txt_blicket_q_{i}",
            index=None,
        )

    # Rule inference
    st.subheader("Rule inference")
    rule_hypothesis = st.text_area(
        "Describe how you think the objects turn on the Nexiom machine.",
        placeholder="e.g. Both object 1 and object 2 need to be on the machine.",
        height=100,
        key="txt_rule_hypothesis",
    )
    rule_type = st.radio(
        "What type of rule do you think applies?",
        ["Conjunctive", "Disjunctive"],
        key="txt_rule_type",
        index=None,
    )

    # Submit
    all_answered = all(blicket_answers.get(f"object_{i}") is not None for i in range(num_objects))
    can_submit = all_answered and rule_type is not None

    if st.button("Submit responses", type="primary", disabled=not can_submit, use_container_width=True):
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_objects": num_objects,
            "num_steps": len(steps),
            "blicket_classifications": blicket_answers,
            "rule_hypothesis": (rule_hypothesis or "").strip(),
            "rule_type": rule_type or "",
        }
        filename = os.path.join(HISTORY_DIR, f"responses_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        st.session_state["txt_last_save"] = filename
        st.success(f"Responses saved to {filename}.")

    if st.session_state.get("txt_last_save"):
        st.caption(f"Last save: {st.session_state['txt_last_save']}")


if __name__ == "__main__":
    main()
