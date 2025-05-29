import os
import json
import random
import datetime

import numpy as np
import streamlit as st

import env.blicket_text as blicket_text

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

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

def start_game():
    """Callback to initialize everything and move to the game phase."""
    env, first_obs = create_new_game(seed=42)
    now = datetime.datetime.now()
    st.session_state.env = env
    st.session_state.start_time = now
    st.session_state.log = [first_obs]
    st.session_state.times = [now]
    st.session_state.phase = "game"

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


BINARY_QUESTIONS = [
    "Did you test each object at least once?",
    "Did you use the feedback from the last test before making a decision?",
    "Were you confident in your final hypothesis?"
]



def submit_qa():
    """Callback when the user clicks ‘Submit Q&A’—writes JSON and goes to end screen."""
    qa_time = datetime.datetime.now()
    binary_answers = {
        question: st.session_state.get(f"qa_{i}", "No")
        for i, question in enumerate(BINARY_QUESTIONS)
    }

    payload = {
        "start_time":     st.session_state.start_time.isoformat(),
        "events": [
            {"time": t.isoformat(), "entry": e}
            for t, e in zip(st.session_state.times, st.session_state.log)
        ],
        "binary_answers": binary_answers,
        "qa_time":        qa_time.isoformat(),
    }

    ts_str   = st.session_state.start_time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOG_DIR, f"game_{ts_str}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # don’t clear everything here—just move to the end‐screen phase
    st.session_state.phase = "end"


def reset_all():
    """Clears all session_state so we go back to the intro screen cleanly."""
    for i in range(len(BINARY_QUESTIONS)):
        st.session_state.pop(f"qa_{i}", None)
    st.session_state.phase      = "intro"
    st.session_state.env        = None
    st.session_state.start_time = None
    st.session_state.log        = []
    st.session_state.times      = []




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

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
st.title("🧙 Text Adventure")

# 1) INTRO SCREEN
if st.session_state.phase == "intro":
    st.markdown(
        """
**Welcome to the Blicket Text Adventure!**

- Test objects (e.g., “test object 1”), press buttons, or ask questions in plain English.  
- Your goal: figure out which objects are the “blickets” by experimenting.

When you’re ready, hit **Start Game**.
"""
    )
    st.button("Start Game", on_click=start_game)
    st.stop()  # halt here until start_game sets phase → "game"

# 2) GAME RUN
elif st.session_state.phase == "game":
    for entry in st.session_state.log:
        st.markdown(f"{entry}")

    st.text_input("What do you do?", key="cmd", on_change=handle_enter)

# 3) Q&A
elif st.session_state.phase == "qa":
    st.markdown("## 📝 Phase 2: Quick Q&A")
    for i, question in enumerate(BINARY_QUESTIONS):
        st.radio(question, ("Yes", "No"), key=f"qa_{i}")
    st.button("Submit Q&A", on_click=submit_qa)

# 4) END-OF-GAME SCREEN
elif st.session_state.phase == "end":
    st.markdown("## 🎉 All done!")
    st.markdown("Thanks for playing—your responses have been saved.")
    st.button("Start Over", on_click=reset_all)
