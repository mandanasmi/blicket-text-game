import os
import json
import random
import datetime
import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import firebase_admin
from firebase_admin import db

# Guard print against BrokenPipeError in Streamlit teardown
import builtins as _builtins

def _safe_print(*args, **kwargs):
    try:
        _builtins.print(*args, **kwargs)
    except BrokenPipeError:
        pass
    except Exception:
        # Swallow any unexpected stdout errors to avoid crashing the app
        pass

print = _safe_print

import env.blicket_text as blicket_text

# Global variable to control visual vs text-only version
USE_TEXT_VERSION = True

def get_image_base64(image_path):
    """Convert image to base64 string for display"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def convert_numpy_types(obj):
    """Convert NumPy types to JSON-serializable Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (set, frozenset)):
        # Convert sets to sorted lists for consistent serialization
        return sorted(convert_numpy_types(list(obj)))
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def convert_to_one_based_indices(indices):
    """Convert 0-based indices to 1-based indices for data storage"""
    if isinstance(indices, (list, np.ndarray)):
        return [i + 1 for i in indices]
    elif isinstance(indices, (int, np.integer)):
        return indices + 1
    else:
        return indices

def convert_to_zero_based_indices(indices):
    """Convert 1-based indices to 0-based indices for internal processing"""
    if isinstance(indices, (list, np.ndarray)):
        return [i - 1 for i in indices]
    elif isinstance(indices, (int, np.integer)):
        return indices - 1
    else:
        return indices

def create_new_game(seed=42, num_objects=4, num_blickets=2, rule="conjunctive", blicket_indices=None):
    """Initialize a fresh BlicketTextEnv and return it plus the first feedback."""
    random.seed(seed)
    np.random.seed(seed)
    env = blicket_text.BlicketTextEnv(
        num_objects=num_objects,
        num_blickets=num_blickets,
        init_prob=0.0,  # Start with all objects OFF the machine
        rule=rule,
        transition_noise=0.0,
        seed=seed,
        blicket_indices=blicket_indices
    )
    game_state = env.reset()
    return env, game_state

def save_game_data(participant_id, game_data):
    """Save game data to Firebase with enhanced tracking"""
    # Convert NumPy types to JSON-serializable types
    game_data = convert_numpy_types(game_data)
    
    try:
        # Get database reference
        db_ref = db.reference()
        participant_ref = db_ref.child(participant_id)
        
        # Determine which key to use based on phase
        phase = game_data.get('phase', 'unknown')
        now = datetime.datetime.now()
        
        if phase == 'main_experiment':
            # For main game: save under main_game/round_X
            games_ref = participant_ref.child('main_game')
            round_number = game_data.get('round_number', 1)
            round_key = f"round_{round_number}"
            games_ref = games_ref.child(round_key)
        elif phase == 'comprehension':
            # For comprehension: save under comprehension key
            games_ref = participant_ref.child('comprehension')
        else:
            games_ref = participant_ref.child('games')  # Fallback
        
        # Enhance game_data with additional metadata
        enhanced_game_data = {
            **game_data,
            "saved_at": now.isoformat(),
            "session_timestamp": now.timestamp()
        }
        
        # For main game rounds, save directly (overwrite if exists)
        # For comprehension, use a specific key
        if phase == 'comprehension':
            print(f"💾 Saving comprehension data to Nexiom database")
            print(f"💾 Saving comprehension data to path: {participant_id}/comprehension")
            print(f"   Data keys: {list(enhanced_game_data.keys())[:10]}...")
            games_ref.set(enhanced_game_data)
            print(f"✅ Successfully saved {phase} data to Nexiom database for {participant_id}")
        else:
            print(f"💾 Saving main game data to Nexiom database")
            print(f"💾 Saving main game data to path: {participant_id}/main_game/{round_key}")
            print(f"   Data keys: {list(enhanced_game_data.keys())[:10]}...")
            games_ref.set(enhanced_game_data)
            print(f"✅ Successfully saved {phase} data to Nexiom database for {participant_id} - Round {round_number}")
    except Exception as e:
        print(f"❌ Failed to save game data for {participant_id}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        print("Game data (not saved):", game_data)

def save_qa_data_immediately(participant_id, round_config, current_round, total_rounds, is_practice, rule_hypothesis, rule_type):
    """Save Q&A data immediately when user provides rule hypothesis"""
    try:
        # Get database reference
        db_ref = db.reference()
        participant_ref = db_ref.child(participant_id)
        
        # Determine which key to use based on phase
        phase = "comprehension" if is_practice else "main_experiment"
        if phase == "main_experiment":
            games_ref = participant_ref.child('main_game')
            entry_id = f"round_{current_round + 1}_qa"
        else:
            games_ref = participant_ref.child('games')
            entry_id = f"comprehension_qa"
        
        # Collect blicket classifications (using 0-based object IDs)
        blicket_classifications = {}
        for i in range(round_config['num_objects']):
            blicket_classifications[f"object_{i}"] = st.session_state.get(f"blicket_q_{i}", "No")
        
        # Calculate total time spent on this round
        end_time = datetime.datetime.now()
        total_time_seconds = (end_time - st.session_state.game_start_time).total_seconds()
        
        # Generate unique round ID
        now = datetime.datetime.now()
        round_id = f"round_{current_round + 1}_{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        
        # Save Q&A data
        user_test_actions_snapshot = st.session_state.user_test_actions.copy() if 'user_test_actions' in st.session_state else []
        qa_data = {
            "round_id": round_id,
            "start_time": st.session_state.game_start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time_seconds": total_time_seconds,
            "round_number": current_round + 1,
            "round_config": round_config,
            "state_history": st.session_state.state_history.copy() if 'state_history' in st.session_state else [],
            "test_timings": st.session_state.get("test_timings", []).copy(),  # Time for each test button press
            "total_actions": len(user_test_actions_snapshot),
            "total_steps_taken": st.session_state.steps_taken if 'steps_taken' in st.session_state else 0,
            "final_machine_state": bool(st.session_state.game_state['true_state'][-1]) if 'game_state' in st.session_state else False,
            "final_objects_on_machine": list(st.session_state.selected_objects) if 'selected_objects' in st.session_state else [],
            "blicket_classifications": blicket_classifications,
            "rule_hypothesis": rule_hypothesis,
            "rule_type": rule_type,
            "true_blicket_indices": round_config.get('blicket_indices', []),
            "true_rule": round_config['rule'],
            "phase": phase,
            "interface_type": "text",
            "qa_saved_immediately": True,
            "qa_submitted_at": now.isoformat()
        }
        
        qa_data["user_test_actions"] = user_test_actions_snapshot
        
        # Convert NumPy types to JSON-serializable types
        qa_data = convert_numpy_types(qa_data)
        
        games_ref.child(entry_id).set(qa_data)
        print(f"✅ Q&A data saved immediately for {participant_id} - Round {current_round + 1}")
        
        # Clear session state and finish round
        reset_game_session_state()
        
        # Return to main app for completion
        if is_practice:
            st.session_state.phase = "practice_complete"
        else:
            st.session_state.phase = "end"
        st.rerun()
        
    except Exception as e:
        print(f"❌ Failed to save Q&A data immediately for {participant_id}: {e}")

def reset_game_session_state():
    """Reset all game-related session state variables"""
    # Clear all game state variables
    game_state_vars = [
        "visual_game_state", "env", "game_state", "object_positions", 
        "selected_objects", "blicket_answers", "game_start_time", 
        "shape_images", "steps_taken", "user_test_actions", "action_history", 
        "state_history", "rule_hypothesis", "rule_type", "test_timings", 
        "previous_test_time"
    ]
    
    for var in game_state_vars:
        st.session_state.pop(var, None)
    
    # Clear blicket question answers
    for i in range(10):  # Clear up to 10 possible blicket questions
        st.session_state.pop(f"blicket_q_{i}", None)
    
    print("🔄 Game session state reset complete")

def save_intermediate_progress(participant_id, round_config, current_round, total_rounds, is_practice=False, blicket_classifications=None, rule_hypothesis=None, rule_type=None, objects_on_machine=None):
    """Save intermediate progress - update single entry with action history and Q&A based on phase"""
    try:
        # Only save intermediate progress for comprehension phase, NOT for main experiment
        if not is_practice:
            print(f"⚠️ Skipping intermediate progress save for main_experiment")
            return
        
        phase = "comprehension"
        
        # Check if Firebase is initialized
        if not firebase_admin._apps:
            print(f"⚠️ Firebase not initialized - skipping {phase} progress save")
            return
        
        # Get database reference
        db_ref = db.reference()
        participant_ref = db_ref.child(participant_id)
        progress_ref = participant_ref.child('comprehension')
        
        # Get existing data or create new
        existing_data = progress_ref.get() or {}
        
        # Update with current action history and Q&A data
        # Note: We don't save round_config here - it's already in the final round data
        now = datetime.datetime.now()
        
        # Create object labels mapping (A/B/C for comprehension)
        label_prefix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        object_labels_mapping = {}
        for obj_idx in range(round_config['num_objects']):
            label = label_prefix[obj_idx]  # A/B/C
            object_labels_mapping[obj_idx] = label
        
        updated_data = {
            **existing_data,
            "last_updated": now.isoformat(),
            "user_test_actions": st.session_state.user_test_actions.copy() if 'user_test_actions' in st.session_state else [],
            "total_actions": len(st.session_state.user_test_actions) if 'user_test_actions' in st.session_state else 0,
            "action_history": st.session_state.action_history.copy() if 'action_history' in st.session_state else [],
            "state_history": st.session_state.state_history.copy() if 'state_history' in st.session_state else [],  # New: Include test history
            "test_timings": st.session_state.get("test_timings", []).copy() if 'test_timings' in st.session_state else [],  # Time for each test button press
            "selected_objects": list(st.session_state.selected_objects) if 'selected_objects' in st.session_state else [],
            "game_start_time": st.session_state.game_start_time.isoformat() if 'game_start_time' in st.session_state else now.isoformat(),
            "phase": phase,
            "round_number": current_round + 1,
            "true_blicket_indices": round_config.get('blicket_indices', []),
            "object_labels_mapping": object_labels_mapping  # New: Object labels mapping
        }
        
        # Add Q&A data if provided
        if blicket_classifications is not None:
            # Map blicket_classifications to labels (A/B/C)
            blicket_classifications_with_labels = {}
            for obj_idx in range(round_config['num_objects']):
                label = label_prefix[obj_idx]
                classification = blicket_classifications.get(f"object_{obj_idx}", "No")
                blicket_classifications_with_labels[label] = {
                    "index": obj_idx,
                    "classification": classification
                }
            updated_data["blicket_classifications"] = blicket_classifications
            updated_data["blicket_classifications_with_labels"] = blicket_classifications_with_labels
        if rule_hypothesis is not None:
            updated_data["rule_hypothesis"] = rule_hypothesis
        if rule_type is not None:
            updated_data["rule_type"] = rule_type
        if objects_on_machine is not None:
            updated_data["objects_on_machine_before_qa"] = objects_on_machine
        
        # Convert NumPy types to JSON-serializable types
        updated_data = convert_numpy_types(updated_data)
        
        progress_ref.set(convert_numpy_types(updated_data))
        print(f"💾 {phase} progress updated for {participant_id} - Round {current_round + 1} - Q&A data included")
        
    except Exception as e:
        import traceback
        print(f"Failed to save {phase} progress for {participant_id}: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

def save_practice_question_answer(participant_id, answer_text):
    """Persist the comprehension practice question response into Firebase."""
    if not answer_text:
        return
    
    try:
        # Ensure Firebase is initialized before attempting to save
        if not firebase_admin._apps:
            print(f"⚠️ Firebase not initialized - skipping practice question save for {participant_id}")
            return
        
        db_ref = db.reference()
        participant_ref = db_ref.child(participant_id)
        progress_ref = participant_ref.child('comprehension')
        existing_data = progress_ref.get() or {}
        now = datetime.datetime.now()
        
        # Attempt to extract a numeric object id from the answer text when possible
        selected_object = None
        lower_text = answer_text.lower()
        if lower_text.startswith("object"):
            try:
                # Expected format: "Object X is a Nexiom"
                selected_object = int(answer_text.split()[1])
            except (ValueError, IndexError):
                selected_object = None
        
        updated_data = {
            **existing_data,
            "practice_blicket_question": {
                "question": "Which object do you think is a Nexiom? **Remember, only Nexioms can turn on the Nexiom machine**.",
                "answer_text": answer_text,
                "selected_object_one_based": selected_object,
                "saved_at": now.isoformat()
            }
        }
        
        progress_ref.set(convert_numpy_types(updated_data))
        print(f"💾 Practice question answer saved for {participant_id}: {answer_text}")
    except Exception as e:
        import traceback
        print(f"Failed to save practice question answer for {participant_id}: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

def textual_blicket_game_page(participant_id, round_config, current_round, total_rounds, save_data_func=None, use_visual_mode=None, is_practice=False):
    """Main blicket game page - text-only interface"""
    
    
    # Determine which mode to use - this is the LOCAL variable for this function
    if use_visual_mode is not None:
        use_text_version = not use_visual_mode
    else:
        # Fall back to global setting or environment variable
        use_text_version = os.getenv('BLICKET_VISUAL_MODE', 'False').lower() != 'true'
    
    # Simple CSS for neutral button styling
    st.markdown("""
    <style>
    /* ===== DESKTOP / DEFAULT BEHAVIOUR ===== */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > div, [data-testid="stMain"] {
        width: 100% !important;
        max-width: 100% !important;
        min-height: 100vh !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .block-container {
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        box-sizing: border-box;
    }

    /* Fixed sidebar width for desktop */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    
    # Initialize or reset session state for this page based on round
    # Check if we need to reinitialize (first time or new round)
    needs_init = "visual_game_state" not in st.session_state
    needs_reset = (st.session_state.get("last_round", -1) != current_round)
    
    if needs_init or needs_reset:
        # Clear visual_game_state to force full reinitialization
        if not needs_init:
            st.session_state.pop("visual_game_state", None)
        
        st.session_state.visual_game_state = "exploration"
        st.session_state.last_round = current_round  # Track which round we're on
        
        st.session_state.env, st.session_state.game_state = create_new_game(
            seed=42 + current_round,
            num_objects=round_config['num_objects'],
            num_blickets=round_config['num_blickets'],
            rule=round_config['rule'],
            blicket_indices=round_config.get('blicket_indices', None)
        )
        st.session_state.object_positions = {}  # Track object positions
        st.session_state.selected_objects = set()  # Objects currently on machine
        st.session_state.blicket_answers = {}  # User's blicket classifications
        st.session_state.game_start_time = datetime.datetime.now()
        st.session_state.steps_taken = 0  # Track number of steps taken
        st.session_state.user_test_actions = []  # Track all user test actions for Firebase
        st.session_state.action_history = []  # Track action history for text version
        st.session_state.state_history = []  # Track complete state history
        st.session_state.test_timings = []  # Track time for each test button press: [{"test_number": 1, "time_seconds": 2.5}, ...]
        st.session_state.previous_test_time = None  # Track timestamp of previous test for interval calculation
        
        print(f"🔄 New round initialized - Round {current_round + 1}/{total_rounds}")
        print(f"   - Objects: {round_config['num_objects']}")
        print(f"   - Nexioms: {round_config['num_blickets']}")
        print(f"   - Rule: {round_config['rule']}")
        print(f"   - Nexiom indices: {round_config.get('blicket_indices', 'None')}")
        
        # Initialize fixed shape images for this round (ensure different images)
        if not use_text_version:
            st.session_state.shape_images = []
            used_shapes = set()
            for i in range(round_config['num_objects']):
                # Keep trying until we get a unique shape
                while True:
                    shape_num = random.randint(1, 8)
                    if shape_num not in used_shapes:
                        used_shapes.add(shape_num)
                        break
                shape_path = f"images/shape{shape_num}.png"
                st.session_state.shape_images.append(get_image_base64(shape_path))
    
    # Get environment state
    env = st.session_state.env
    game_state = st.session_state.game_state
    
    
    # Load images
    blicket_img = get_image_base64("images/blicket.png")
    blicket_lit_img = get_image_base64("images/blicket_lit.png")
    
    # Initialize shape images for visual version
    shape_images = []
    if not use_text_version:
        if "shape_images" not in st.session_state:
            # Initialize shape images if not already done (ensure different images)
            st.session_state.shape_images = []
            used_shapes = set()
            for i in range(round_config['num_objects']):
                # Keep trying until we get a unique shape
                while True:
                    shape_num = random.randint(1, 8)
                    if shape_num not in used_shapes:
                        used_shapes.add(shape_num)
                        break
                shape_path = f"images/shape{shape_num}.png"
                st.session_state.shape_images.append(get_image_base64(shape_path))
        
        shape_images = st.session_state.shape_images
        
        # Pre-load all shape images for state history (smaller versions)
        if "shape_images_small" not in st.session_state:
            st.session_state.shape_images_small = []
            for i in range(round_config['num_objects']):
                shape_num = (i % 8) + 1  # Use consistent mapping
                shape_path = f"images/shape{shape_num}.png"
                st.session_state.shape_images_small.append(get_image_base64(shape_path))
        
        shape_images_small = st.session_state.shape_images_small
    
    # Create sidebar for state history - matching right sidebar style
    with st.sidebar:
        st.markdown("""
        <div style="background: #0d47a1; padding: 12px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #0b3779; color: #ffffff; text-align: center; font-size: 16px; font-weight: bold;">
            Test History
        </div>
        """, unsafe_allow_html=True)
        
        history_container = st.container()
        with history_container:
            if st.session_state.state_history:
                total_tests = len(st.session_state.state_history)
                max_tests = 5 if is_practice else 16
                st.markdown(
                    f"<div style='text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 12px; padding: 10px; background-color: #555555; color: #ffffff; border-radius: 6px; border: 1px solid #3a3a3a;'>Total Test: {total_tests} out of {max_tests}</div>",
                    unsafe_allow_html=True,
                )
                
                for i, state in enumerate(st.session_state.state_history):
                    if use_text_version:
                        # Text version: show object yes/no format with ID on top
                        objects_text = ""
                        for obj_idx in range(round_config['num_objects']):
                            label_prefix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if is_practice else "1234567890"
                            object_id = obj_idx  # 0-based object ID
                            display_id = label_prefix[obj_idx] if is_practice else object_id + 1  # 1-based or letter
                            is_on_platform = object_id in state['objects_on_machine']
                            yes_no = "Yes" if is_on_platform else "No"
                            # Yes = white background, No = gray background (always consistent)
                            bg_color = "#ffffff" if is_on_platform else "#d0d0d0"  # white for Yes, gray for No
                            border_style = "2px solid #999" if is_on_platform else "1px solid #ccc"
                            objects_text += (
                                "<span style='display: inline-flex; align-items: center; justify-content: center; "
                                f"background-color: {bg_color}; color: black; padding: 3px 5px; margin: 2px 2px; border-radius: 4px; "
                                f"font-size: 16px; font-weight: bold; border: {border_style}; min-width: 48px; flex-shrink: 0;'>"
                                f"<div style='font-size: 16px; margin-right: 3px; font-weight: bold;'>{display_id}</div>"
                                f"<div style='font-size: 16px; font-weight: bold;'>{yes_no}</div></span>"
                            )
                        
                        # Show machine state on same row
                        machine_status = "ON" if state['machine_lit'] else "OFF"
                        machine_color = "#388e3c" if state['machine_lit'] else "#333333"
                        st.markdown(
                            f"""
                        <div style='
                            width: 100%;
                            margin: 8px 0;
                            padding: 12px 16px;
                            background-color: #a9a9a9;
                            color: #1f1f1f;
                            border: 1px solid #7f7f7f;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
                            box-sizing: border-box;
                        '>
                            <div style='font-size: 16px; font-weight: bold; margin-bottom: 6px;'>Test {i + 1}</div>
                            <div style='margin-bottom: 6px; font-size: 16px; display: flex; flex-wrap: wrap; justify-content: center; gap: 6px;'>{objects_text}</div>
                            <div style='font-size: 16px; font-weight: bold; color: {("#79ff4d" if state["machine_lit"] else "#000000")}'>Machine: {machine_status}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Visual version: show object numbers with colored backgrounds
                        cols = st.columns(round_config['num_objects'] + 2)
                        
                        with cols[0]:
                            st.markdown(f"<div style='font-size: 16px; font-weight: bold; margin: 8px 0;'>Test {i + 1}</div>", unsafe_allow_html=True)
                        
                        # Show each object
                        for obj_idx in range(round_config['num_objects']):
                            object_id = obj_idx
                            display_label = label_prefix[obj_idx] if is_practice else obj_idx + 1
                            with cols[obj_idx + 1]:
                                is_on_platform = object_id in state['objects_on_machine']
                                yes_no = "Yes" if is_on_platform else "No"
                                # Yes = white background, No = gray background (always consistent)
                                bg_color = "#ffffff" if is_on_platform else "#d0d0d0"  # white for Yes, gray for No
                                border_style = "2px solid #999" if is_on_platform else "1px solid #ccc"
                                st.markdown(
                                    f"""
                                    <div style="
                                    background-color: {bg_color}; 
                                    border: {border_style}; 
                                    border-radius: 5px; 
                                    padding: 8px; 
                                    margin: 2px; 
                                        text-align: center;
                                        color: black;
                                        font-weight: bold;
                                    min-width: 45px;
                                    ">
                                         <div style="font-size: 16px; margin-bottom: 4px; color: #333;">
                                        {display_label}
                                    </div>
                                         <div style="font-size: 16px;">
                                        {yes_no}
                                    </div>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                        
                        # Show machine status
                        with cols[-1]:
                            machine_status = "ON" if state['machine_lit'] else "OFF"
                            machine_color = "#000000"  # Always black
                            st.markdown(f"<div style='font-size: 16px; margin: 8px 0; font-weight: bold; color: {machine_color};'>{machine_status}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                <div style='
                    padding: 20px; 
                    text-align: center; 
                    background-color: #f0f0f0; 
                    border-radius: 5px; 
                    color: #666;
                    font-size: 16px;
                '>
                No tests recorded yet. Click "Test Machine" to begin.
                </div>
                """,
                    unsafe_allow_html=True,
                )

    
    # Main content area
    # Display round info and progress
    st.markdown(f"<div style='font-size: 20px; font-weight: bold; margin-bottom: 4px;'>Round {current_round + 1} of {total_rounds}</div>", unsafe_allow_html=True)
    progress = (current_round + 1) / total_rounds
    st.markdown(
        f"<div style='width: 100%; height: 10px; background: #d0e2ff; border-radius: 999px; overflow: hidden; margin-bottom: 16px;'>"
        f"<div style='width: {progress * 100}%; height: 100%; background: #0d47a1;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    
    # Collapsible instruction section
    with st.expander("Click to read game instructions", expanded=False):
        horizon = round_config.get('horizon', 32)  # Default to 32 tests
        st.markdown(f"""

        **Your goals are:**
        - Identify which objects will turn on the machine. **It could be one or multiple objects.**
        - Infer the underlying rule for how the machine turns on. 

        **Tips:**
        - All objects can be either on the machine or on the floor.
        - You should think about how to efficiently explore the relationship between the objects and the machine.

        You have **{horizon} tests** to complete the task. You can also exit the task early if you think you understand the relationship between the objects and the machine. After the task is done, you will be asked which objects are Nexioms, and the rule for turning on the machine.

        You will be prompted at each turn to choose tests.

        **Understanding the Test History:**
        - The **Test History** panel on the left shows a record of each test you perform.
        - For each object, it shows **Yes** if the object was **ON the platform**, or **No** if it was **NOT on the platform**.
        - Each row also shows whether the machine was **ON** or **OFF** after that test.

        **How to use the interface:**
        1. Click on object buttons to **select** the objects you want to test
        2. A status box below each button shows: **ON PLATFORM** or **NOT ON PLATFORM**
        3. Click an object again to **deselect** it
        4. Once you have selected your combination, click the **Test Machine** button
        5. The test will be recorded, and you'll see the result in the Test History
        6. Repeat: select new objects and test again as needed
        """)

    

    
    # Decide label prefix (letters for practice, numbers otherwise)
    label_prefix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if is_practice else "1234567890"
    
    # Text-only version: Display action history
    if use_text_version:
        st.markdown("<div style='font-size: 30px !important; font-weight: 700; margin-bottom: 0.5rem;'>Action History</div>", unsafe_allow_html=True)
        if st.session_state.action_history:
            # Limit visible entries to roughly eight before scrolling
            def format_action(entry: str) -> str:
                # Labels are already correct (A/B/C for comprehension, 1/2/3/4 for main game)
                return f"<div style='font-size: 16px; margin-bottom: 0.15rem;'>• {entry}</div>"
            entries_html = "".join(format_action(action_text) for action_text in st.session_state.action_history)
            st.markdown(
                f"""
                <div style='max-height: 12.5rem; overflow-y: auto; padding-right: 18px; margin-bottom: 0.5rem; border: 2px solid #0d47a1; border-radius: 10px; padding: 0.5rem 0.2rem; box-shadow: inset 0 0 6px rgba(0,0,0,0.08); direction: rtl;'>
                    <div style='direction: ltr; padding: 0 0.9rem 0 0.65rem;'>
                        {entries_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("*No actions taken yet.*")
        st.markdown("---")
    
    # Display the Nexiom machine (only in visual version)
    if not use_text_version:
        st.markdown("### The Nexiom Machine")
    
    # Determine if machine should be lit
    machine_lit = game_state['true_state'][-1]
    
    if not use_text_version:
        machine_img = blicket_lit_img if machine_lit else blicket_img
        
        # Create machine display with steps counter
        horizon = round_config.get('horizon', 32)
        steps_left = horizon - st.session_state.steps_taken
        
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
                <div>
                    <img src="data:image/png;base64,{machine_img}" style="width: 200px; height: auto;">
                    <div style="margin-top: 10px; font-size: 18px; font-weight: bold; color: #333;">
                        Nexiom Machine: {'ON' if machine_lit else 'OFF'}
                    </div>
                </div>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px 25px; border-radius: 15px; color: white; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                    <div style="font-size: 14px; margin-bottom: 5px;">Number of Remaining Tests</div>
                    <div style="font-size: 24px;">{steps_left}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Text-only version: Show steps counter and machine status
    if use_text_version:
        horizon = round_config.get('horizon', 32)
        steps_left = horizon - st.session_state.steps_taken
    
    

    
    # Display available objects
    practice_locked = is_practice and st.session_state.get("show_practice_test", False)
    qna_locked = (st.session_state.visual_game_state != "exploration") or practice_locked
    
    if use_text_version:
        header = "Available Objects (A, B, C)" if is_practice else "Available Objects"
        st.markdown(f"<div style='font-size: 30px !important; font-weight: 700; margin-bottom: 0.5rem;'>{header}</div>", unsafe_allow_html=True)
        
        # Text-only version: Simple button grid with selection mode
        st.markdown('<div class="object-grid-wrapper comprehension-layout">', unsafe_allow_html=True)
        cols = st.columns(2, gap="large") if st.session_state.get("screen_is_small") else st.columns(4, gap="medium")
        for i in range(round_config['num_objects']):
            col_index = i % (2 if st.session_state.get("screen_is_small") else 4)
            with cols[col_index]:
                object_id = i  # 0-based object ID
                display_label = label_prefix[i] if is_practice else f"{i + 1}"
                is_selected = object_id in st.session_state.selected_objects
                horizon = round_config.get('horizon', 32)
                steps_left = horizon - st.session_state.steps_taken
                interaction_disabled = (steps_left <= 0 or st.session_state.visual_game_state != "exploration") or qna_locked
                
                # Use neutral styling - status will be shown in box below
                # Wrap button + status box for styling
                container_title = "Interaction disabled during Q&A." if qna_locked else ""
                st.markdown(
                    f'<div style="display: flex; flex-direction: column; align-items: center; gap: 0.35rem; width: 100%; max-width: 210px; min-width: 0;" title="{container_title}">',
                    unsafe_allow_html=True,
                )
                button_help = "Interaction disabled during Q&A." if qna_locked else f"Click to {'remove' if is_selected else 'place'} Object {display_label}"
                button_clicked = st.button(
                    f"Object {display_label}",
                           key=f"obj_{i}", 
                           disabled=interaction_disabled,
                    help=button_help,
                    type="secondary",
                )

                if button_clicked:
                    # Toggle selection and record action
                    if is_selected:
                        st.session_state.selected_objects.remove(object_id)
                        action_text = f"Removed Object {display_label} from machine"
                    else:
                        st.session_state.selected_objects.add(object_id)
                        action_text = f"Placed Object {display_label} on machine"
                    
                    # Add to action history
                    st.session_state.action_history.append(action_text)
                    
                    st.rerun()
                
                # Show status box underneath button
                status_text = "ON PLATFORM" if is_selected else "NOT ON PLATFORM"
                box_border_color = "#4a4a4a"
                status_color = "#388e3c" if is_selected else "#333"  # Green when on platform, dark gray otherwise
                background_color = "#dfeee0" if is_selected else "#f5f5f5"  # Slightly lighter background shades
                st.markdown(
                    f"""
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    width: 100%;
                    max-width: 210px;
                    min-width: 0;
                    min-height: 50px;
                    padding: clamp(6px, 0.8vw, 10px);
                    border: 1px solid {box_border_color};
                    border-radius: 10px;
                    background-color: {background_color};
                    font-size: clamp(15px, 1.3vw, 19px);
                    color: {status_color};
                    font-weight: 700;
                ">{status_text}</div>
                """,
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # Show the Test Machine button after object selection area
        st.markdown("---")        
        current_selection = list(st.session_state.selected_objects)
        if steps_left <= 0:
            st.markdown("<p style='color: #d32f2f; font-weight: 700; margin-bottom: 0.5rem;'>No Tests remaining! You need to answer the question below.</p>", unsafe_allow_html=True)

        if current_selection:
            selection_text = ", ".join(
                [f"Object {label_prefix[obj] if is_practice else obj + 1}" for obj in sorted(current_selection)]
            )
            st.markdown(f"<strong>Current selection:</strong> {selection_text}", unsafe_allow_html=True)
        else:
            st.markdown("**No objects selected yet.** Click on objects to select them.")
        
        # Test Machine button - only appears if objects are selected
        test_help = "Interaction disabled during Q&A." if qna_locked else None
        if st.button("Test Machine", type="primary", disabled=not current_selection or interaction_disabled, help=test_help):
            # NOW record the test action
            action_time = datetime.datetime.now()
            
            # Calculate time for this test
            test_number = st.session_state.steps_taken + 1
            if st.session_state.previous_test_time is None:
                # First test: time since game start
                time_since_start = (action_time - st.session_state.game_start_time).total_seconds()
                time_since_previous = time_since_start  # Same as time since start for first test
            else:
                # Subsequent tests: time since previous test
                time_since_previous = (action_time - st.session_state.previous_test_time).total_seconds()
                time_since_start = (action_time - st.session_state.game_start_time).total_seconds()
            
            # Store test timing
            test_timing = {
                "test_number": test_number,
                "time_since_start_seconds": round(time_since_start, 3),  # Time from game start to this test
                "time_since_previous_seconds": round(time_since_previous, 3),  # Time from previous test to this test
                "timestamp": action_time.isoformat()
            }
            if "test_timings" not in st.session_state:
                st.session_state.test_timings = []
            st.session_state.test_timings.append(test_timing)
            st.session_state.previous_test_time = action_time
            
            # Capture machine state BEFORE this test
            machine_state_before = bool(game_state['true_state'][-1]) if 'game_state' in st.session_state else False
            
            # Update environment state with current object selections
            for idx in range(round_config['num_objects']):
                env._state[idx] = (idx in st.session_state.selected_objects)
            env._update_machine_state()
            game_state = env.step("look")[0]
            st.session_state.game_state = game_state
                    
            # Get final result of this test combination
            final_machine_state = bool(game_state['true_state'][-1])
            
            # Add to action history - show test result
            action_text = f"Test Result: Nexiom machine is {'ON' if final_machine_state else 'OFF'}"
            st.session_state.action_history.append(action_text)
                    
            # Add to state history (only on Test Machine button click)
            # Create object labels mapping (A/B/C for comprehension, 1/2/3/4 for main game)
            label_prefix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if is_practice else "1234567890"
            objects_with_labels = {}
            for obj_idx in range(round_config['num_objects']):
                object_id = obj_idx  # 0-based
                label = label_prefix[obj_idx] if is_practice else str(object_id + 1)  # A/B/C or 1/2/3/4
                is_on_machine = object_id in st.session_state.selected_objects
                objects_with_labels[label] = {
                    "index": object_id,
                    "on_machine": is_on_machine,
                    "status": "Yes" if is_on_machine else "No"
                }
            
            state_data = {
                "objects_on_machine": set(st.session_state.selected_objects),  # Keep for compatibility (0-based indices)
                "objects_with_labels": objects_with_labels,  # New: object labels with status
                "machine_lit": final_machine_state,
                "step_number": st.session_state.steps_taken + 1
            }
            st.session_state.state_history.append(state_data)
                    
            # Record the action for Firebase
            # Create objects_tested with labels
            objects_tested_with_labels = {}
            for obj_idx in st.session_state.selected_objects:
                label = label_prefix[obj_idx] if is_practice else str(obj_idx + 1)
                objects_tested_with_labels[label] = obj_idx  # Store label -> index mapping
            
            action_data = {
                "timestamp": action_time.isoformat(),
                "action_type": "test",
                "objects_tested": list(st.session_state.selected_objects),  # Keep for compatibility (0-based indices)
                "objects_tested_with_labels": objects_tested_with_labels,  # New: object labels
                "machine_state_before": machine_state_before,
                "machine_state_after": final_machine_state,
                "step_number": st.session_state.steps_taken + 1
            }
            st.session_state.user_test_actions.append(action_data)
                    
            # Increment step counter only on test
            st.session_state.steps_taken += 1
                    
            # Save intermediate progress after each test (only for comprehension phase)
            if is_practice:
                save_intermediate_progress(participant_id, round_config, current_round, total_rounds, is_practice)
            
            st.rerun()
        # Show Nexiom Machine Status below Test Combination button (only after at least one test)
        if st.session_state.state_history:
            machine_status = "ON" if machine_lit else "OFF"
            status_color = "#66bb6a" if machine_lit else "#000000"  # Green when ON, black when OFF
            st.markdown(
                f"""
                <div style='font-size: 20px; font-weight: 700; margin-bottom: 0.35rem;'>
                    Nexiom Machine Status: <span style='color: {status_color};'>{machine_status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    else:
        instructions = "Interaction is disabled during Q&A." if qna_locked else "Click on an object to place it on the machine. Click again to remove it."
        st.markdown(instructions)
        
        # Visual version: Create grid of objects with clickable images
        for i in range(0, round_config['num_objects'], 4):
            # Create a row of up to 4 objects
            row_objects = range(i, min(i + 4, round_config['num_objects']))
        
            # Always create 4 columns for consistent layout
            cols = st.columns(4)
            
            for j in range(4):
                with cols[j]:
                    # Check if this column should have an object
                    if j < len(row_objects):
                        obj_idx = row_objects[j] + 1  # Convert to 1-based object ID
                    else:
                        # Empty column - skip rendering
                        continue
                    
                    # Check if object is currently selected
                    is_selected = obj_idx in st.session_state.selected_objects
                    horizon = round_config.get('horizon', 32)
                    steps_left = horizon - st.session_state.steps_taken
                    
                    # Disable interaction if no steps left or if in questionnaire phase
                    interaction_disabled = (steps_left <= 0 or st.session_state.visual_game_state != "exploration") or qna_locked
                    
                    # Create clickable image with improved styling (neutral colors, no green)
                    if interaction_disabled:
                        # Disabled state - gray out the image
                        opacity = "0.5"
                        cursor = "not-allowed"
                        border_color = "#cccccc"
                    else:
                        opacity = "1.0"
                        cursor = "pointer"
                        border_color = "#666666" if is_selected else "#ffffff"
                    
                    # Create clickable image container
                    if interaction_disabled:
                        # Disabled state - just show the image
                        st.markdown(f"""
                        <div style="text-align: center; margin: 10px;">
                            <div style="
                                display: inline-block; 
                                padding: 15px; 
                                border: 3px solid {border_color}; 
                                border-radius: 15px; 
                                background: {'#e8e8e8' if is_selected else 'transparent'};
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                                opacity: {opacity};
                            ">
                                <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 80px; height: auto; margin-bottom: 10px;">
                                <br>
                                <div style="font-weight: bold; color: #333; font-size: 16px;">
                                    Object {obj_idx + 1}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Interactive state - make the entire image container clickable
                        
                                                # Create the image display
                        st.markdown(f"""
                        <div style="margin: 10px;">
                            <div style="
                                display: inline-block; 
                                padding: 15px; 
                                border: 3px solid {border_color}; 
                                border-radius: 15px; 
                                background: {'#d0d0d0' if is_selected else 'transparent'};
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                                transition: all 0.2s ease;
                                position: relative;
                            " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 6px 20px rgba(0,0,0,0.15)'" onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.1)'">
                                <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 80px; height: auto; margin-bottom: 10px;">
                                <br>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Select Object {obj_idx + 1}", key=f"obj_{obj_idx}", help=f"Click to {'remove' if is_selected else 'place'} Object {obj_idx + 1}"):
                            # Record the action before making changes
                            action_time = datetime.datetime.now()
                            action_type = "remove" if is_selected else "place"
                            
                            # Capture machine state BEFORE making changes
                            machine_state_before = bool(game_state['true_state'][-1]) if 'game_state' in st.session_state else False
                            
                            # Update object selection
                            if is_selected:
                                st.session_state.selected_objects.remove(obj_idx)
                            else:
                                st.session_state.selected_objects.add(obj_idx)
                            
                            # Update environment state (convert to 0-based for internal state)
                            env._state[obj_idx - 1] = (obj_idx in st.session_state.selected_objects)
                            env._update_machine_state()
                            game_state = env.step("look")[0]  # Get updated state
                            st.session_state.game_state = game_state
                            
                            # Add to state history
                            # Create object labels mapping (A/B/C for comprehension, 1/2/3/4 for main game)
                            label_prefix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if is_practice else "1234567890"
                            objects_with_labels = {}
                            for obj_idx_range in range(round_config['num_objects']):
                                object_id = obj_idx_range  # 0-based
                                label = label_prefix[obj_idx_range] if is_practice else str(object_id + 1)  # A/B/C or 1/2/3/4
                                is_on_machine = object_id in st.session_state.selected_objects
                                objects_with_labels[label] = {
                                    "index": object_id,
                                    "on_machine": is_on_machine,
                                    "status": "Yes" if is_on_machine else "No"
                                }
                            
                            state_data = {
                                "objects_on_machine": set(st.session_state.selected_objects),  # Keep for compatibility (0-based indices)
                                "objects_with_labels": objects_with_labels,  # New: object labels with status
                                "machine_lit": bool(game_state['true_state'][-1]),
                                "step_number": st.session_state.steps_taken + 1
                            }
                            st.session_state.state_history.append(state_data)
                            
                            display_label = label_prefix[obj_idx] if is_practice else f"{obj_idx + 1}"
                            selection_text = f"{'Removed' if is_selected else 'Placed'} Object {display_label} on machine"
                            st.session_state.action_history.append(selection_text)

                            # Record the action for Firebase
                            # Create objects with labels for action
                            obj_label = label_prefix[obj_idx - 1] if is_practice else str(obj_idx)  # obj_idx is 1-based in visual mode
                            action_data = {
                                "timestamp": action_time.isoformat(),
                                "action_type": action_type,
                                "object_index": obj_idx - 1,  # Convert to 0-based for consistency
                                "object_label": obj_label,  # New: object label (A/B/C or 1/2/3/4)
                                "object_id": f"object_{obj_idx}",
                                "machine_state_before": machine_state_before,  # Machine state before this action
                                "machine_state_after": bool(game_state['true_state'][-1]),  # New state
                                "objects_on_machine": list(st.session_state.selected_objects),
                                "step_number": st.session_state.steps_taken + 1
                            }
                            st.session_state.user_test_actions.append(action_data)
                            
                            # Increment step counter
                            st.session_state.steps_taken += 1
                            
                            # Save intermediate progress after each action (only for comprehension phase)
                            if is_practice:
                                save_intermediate_progress(participant_id, round_config, current_round, total_rounds, is_practice)
                            
                            st.rerun()
                    
                    
    

    
    # Phase transition buttons
    
    if st.session_state.visual_game_state == "exploration":
        horizon = round_config.get('horizon', 32)
        steps_left = horizon - st.session_state.steps_taken
        
        # Show different buttons based on steps remaining
        if steps_left <= 0:
            if is_practice:                # Comprehension phase - show practice test on same page
                st.markdown("---")
                st.markdown("### Practice Q&A")
                st.markdown("Which object do you think is a Nexiom? Remember, only Nexioms can turn on the Nexiom machine.")
                
                # Create options for the practice test
                num_objects_in_round = round_config['num_objects']
                practice_options = [
                    f"Object {label_prefix[i]} is a Nexiom" if is_practice else f"Object {i + 1} is a Nexiom"
                    for i in range(num_objects_in_round)
                ]
                practice_options.append("I don't know")
                
                # Show single question with all options
                practice_answer = st.radio(
                    "Select your answer:",
                    practice_options,
                    key="practice_blicket_answer_inline",
                    index=None
                )
                
                if st.button("Complete Comprehension Phase", type="primary", disabled=(practice_answer is None)):
                    # Save response before moving on
                    save_practice_question_answer(participant_id, practice_answer)
                    # Move to practice completion page
                    st.session_state.phase = "practice_complete"
                    st.rerun()
            else:
                # Main experiment - show questionnaire
                st.markdown("""
                <div style="background: rgba(255, 193, 7, 0.8); border: 2px solid #ffc107; border-radius: 10px; padding: 20px; margin: 20px 0; text-align: center;">
                    <h4 style="color: #856404; margin: 10px 0;">Please proceed to answer questions about which objects are Nexioms.</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("PROCEED TO ANSWER QUESTIONS", type="primary", key="proceed_btn"):
                    # Clear any previous answers to ensure fresh start
                    for i in range(round_config['num_objects']):
                        if f"blicket_q_{i}" in st.session_state:
                            del st.session_state[f"blicket_q_{i}"]
                    # Clear rule inference and rule type answers
                    if "rule_hypothesis" in st.session_state:
                        del st.session_state["rule_hypothesis"]
                    if "rule_type" in st.session_state:
                        del st.session_state["rule_type"]
                    st.session_state.visual_game_state = "questionnaire"
                    st.rerun()
        else:
            # Check if practice Q&A has started
            practice_qa_started = is_practice and st.session_state.get("show_practice_test", False)
            
            # Check if participant has tested at least once or twice (to ensure they've explored)
            num_tests = len(st.session_state.get('state_history', []))
            has_tested_at_least_once = num_tests >= 1
            
            # Only show message box if participant has tested at least once
            if has_tested_at_least_once:
                # Update message based on whether Q&A has started
                if practice_qa_started:
                    message_text = "Once you press 'Ready to Answer Questions', you cannot go back to explore objects and test the machine."
                else:
                    if is_practice:
                        message_text = "You can continue testing the machine or hit Ready to Answer Questions to answer questions about Nexioms. <strong>Note: Once you press 'Ready to Answer Questions', you cannot go back to explore objects and test the machine.</strong>"
                    else:
                        message_text = "You can continue testing the machine or hit Ready to Answer Questions to answer questions about Nexioms. <strong>Note: Once you press 'Ready to Answer Questions', you cannot go back to explore objects and test the machine.</strong>"
                
                st.markdown(f"""
                    <div style="background: rgba(29, 161, 242, 0.1); border: 2px solid #1DA1F2; border-radius: 10px; padding: 20px; margin: 20px 0; text-align: center;">
                        <h3 style="color: #1DA1F2; margin: 0;"> You have {steps_left} Tests remaining</h3>
                        <p style="color: #1DA1F2; margin: 10px 0;">{message_text}</p>
                    </div>
                """, unsafe_allow_html=True)
            
                # Show different button based on phase (only after at least one test)
                if is_practice:
                    # Comprehension phase - show practice test inline
                    if st.button("Ready to Answer Questions", type="primary", key="complete_ready_btn"):
                        # Set flag to show practice test inline
                        st.session_state.show_practice_test = True
                        st.rerun()
                else:
                    # Main experiment - show questionnaire button
                    if st.button("Ready to Answer Questions", type="primary", key="ready_btn"):
                        # Clear any previous Nexiom answers to ensure fresh start
                        for i in range(round_config['num_objects']):
                            if f"blicket_q_{i}" in st.session_state:
                                del st.session_state[f"blicket_q_{i}"]
                        st.session_state.visual_game_state = "questionnaire"
                        st.rerun()
            
            
            # Show practice test inline if flag is set
            if is_practice and st.session_state.get("show_practice_test", False):
                st.markdown("---")
                st.markdown("### Practice Q&A")
                st.markdown("Which object do you think is a Nexiom? Remember, only Nexioms can turn on the Nexiom machine.")
                
                # Create options for the practice test
                num_objects_in_round = round_config['num_objects']
                practice_options = [
                    f"Object {label_prefix[i]} is a Nexiom" if is_practice else f"Object {i + 1} is a Nexiom"
                    for i in range(num_objects_in_round)
                ]
                practice_options.append("I don't know")
                
                # Show single question with all options
                practice_answer = st.radio(
                    "Select your answer:",
                    practice_options,
                    key="practice_blicket_answer_steps_remaining",
                    index=None
                )
                
                if st.button("Complete Comprehension Phase", type="primary", disabled=(practice_answer is None), key="complete_from_steps"):
                    # Save response before moving on
                    save_practice_question_answer(participant_id, practice_answer)
                    # Move to practice completion page
                    st.session_state.phase = "practice_complete"
                    st.session_state.pop("show_practice_test", None)
                    st.rerun()
    
    elif st.session_state.visual_game_state == "questionnaire" and not is_practice:
        st.markdown("""
        <div style="padding: 20px; border-radius: 15px; background-color: #e3f2fd; border: 2px solid #2196f3; margin: 20px 0;">
            <h3 style="margin: 0; text-align: center; color: #1565c0;">🎯 Nexiom Classification</h3>
            <p style="margin: 10px 0 0 0; text-align: center; color: #1976d2;">For each object, indicate whether you think it is a Nexiom or not:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Debug: Check existing session state before rendering radio buttons
        print(f"🔍 DEBUG: Rendering questionnaire, checking session state...")
        for i in range(round_config['num_objects']):
            key = f"blicket_q_{i}"
            if key in st.session_state:
                print(f"   - {key} already exists: {st.session_state.get(key)}")
        
        # Create questionnaire with object images
        for i in range(round_config['num_objects']):
            if use_text_version:
                # Text-only version: Simple text-based questionnaire
                st.radio(
                    f"Is Object {i + 1} a Nexiom?",
                    ["Yes", "No"],
                    key=f"blicket_q_{i}",
                    index=None
                )
            else:
                # Visual version: Questionnaire with images
                st.markdown(f"""
                <div style="display: inline-flex; align-items: center; margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <div style="flex: 0 0 100px; text-align: center;">
                        <img src="data:image/png;base64,{shape_images[i]}" style="width: 60px; height: auto;">
                        <br><strong>Object {i + 1}</strong>
                    </div>
                    <div style="flex: 0 0 auto; margin-left: 20px;">
                """, unsafe_allow_html=True)
                
                st.radio(
                    f"Is Object {i + 1} a Nexiom?",
                    ["Yes", "No"],
                    key=f"blicket_q_{i}",
                    index=None
                )
                
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Add rule question
        st.markdown("---")
        st.markdown("### Rule Inference")
        st.markdown("Based on your observations for this round only, describe how the objects turn on this Nexiom machine.")
        rule_input_value = st.text_area(
            "What do you think is the rule?",
            placeholder="Describe your hypothesis about how the Nexiom machine determines when to switch on...",
            height=100,
            key="rule_hypothesis"
        )
        
        # Navigation buttons - allow users to save and continue or finish
        st.markdown("---")
        st.markdown("### Continue to Rule Type Classification")
        
        # Check if all blicket questions are answered
        all_blicket_answered = True
        for i in range(round_config['num_objects']):
            if st.session_state.get(f"blicket_q_{i}", None) is None:
                all_blicket_answered = False
                break
        
                # Check if rule hypothesis is provided
        current_hypothesis = rule_input_value.strip()
        
        # Determine what is still missing and show inline guidance
        missing_messages = []
        if not all_blicket_answered:
            missing_messages.append("Please answer all Nexiom questions.")
        if not current_hypothesis:
            missing_messages.append("Please enter a rule hypothesis.")
        
        button_disabled = bool(missing_messages)
        next_button_clicked = st.button(
            "NEXT: Rule Type Classification",
            type="primary",
            use_container_width=True,
            disabled=button_disabled
        )
        
        for msg in missing_messages:
            st.markdown(f"<p style='color: #dc3545; font-size: 14px;'>{msg}</p>", unsafe_allow_html=True)
        
        if next_button_clicked and not button_disabled:
            # All validations passed
            print(f"🔍 DEBUG: Raw rule_hypothesis from widget: '{st.session_state.get('rule_hypothesis', 'NOT FOUND')}'")
            print(f"🔍 DEBUG: Trimmed current_hypothesis: '{current_hypothesis}'")
                
            # Save Nexiom classifications before moving to rule type
            blicket_classifications = {}
            print(f"🔍 DEBUG: About to collect Nexiom answers, num_objects = {round_config['num_objects']}")
            for i in range(round_config['num_objects']):
                raw_answer = st.session_state.get(f"blicket_q_{i}", None)
                print(f"🔍 DEBUG: Raw blicket_q_{i} from session state: {raw_answer}")
                # Only save actual values from the user, don't default to "No"
                if raw_answer is not None:
                    blicket_classifications[f"object_{i}"] = raw_answer
                    print(f"🔍 Saving intermediate - blicket_q_{i} = {raw_answer}")
                else:
                    print(f"⚠️  WARNING: blicket_q_{i} is None - user didn't select an answer!")
                    
            # Get objects that were on the machine before Q&A
            objects_on_machine_before_qa = list(st.session_state.get("selected_objects", set()))
                    
            # Note: We don't save intermediate progress for main_game rounds anymore
            # All data including hypothesis and rule_type will be saved in the final round data
            # Only save for comprehension phase if needed
            if is_practice:
                save_intermediate_progress(
                    participant_id, 
                    round_config, 
                    current_round, 
                    total_rounds, 
                    is_practice,
                    blicket_classifications=blicket_classifications,
                    rule_hypothesis=current_hypothesis,
                    objects_on_machine=objects_on_machine_before_qa
                )
                print("✅ Saved intermediate progress for comprehension phase")
                    
            print(f"📝 Preparing to save hypothesis for round {current_round + 1}")
            print(f"   - Objects on machine: {objects_on_machine_before_qa}")
            print(f"   - Nexiom classifications: {blicket_classifications}")
            print(f"   - Hypothesis: {current_hypothesis[:50]}...")
                    
            # Debug: Check session state before transition
            print("🔍 DEBUG: Before transitioning to rule_type, checking session state:")
            print(f"   - rule_hypothesis: {st.session_state.get('rule_hypothesis', 'NOT FOUND')}")
            for i in range(round_config['num_objects']):
                key = f"blicket_q_{i}"
                if key in st.session_state:
                    print(f"   - {key}: {st.session_state.get(key, 'NOT FOUND')}")
                    
            # Persist data needed for rule-type classification stage
            st.session_state["saved_blicket_classifications"] = blicket_classifications
            st.session_state["saved_rule_hypothesis"] = current_hypothesis
                    
            st.session_state.visual_game_state = "rule_type_classification"
            st.rerun()


    elif st.session_state.visual_game_state == "rule_type_classification" and not is_practice:
        st.markdown("""
        <div style="padding: 20px; border-radius: 15px; background-color: #f3e5f5; border: 2px solid #9c27b0; margin: 20px 0;">
            <h3 style="margin: 0; text-align: center; color: #4a148c;">🎯 Rule Type Classification</h3>
            <p style="margin: 10px 0 0 0; text-align: center; color: #6a1b9a;">Based on your observations, what type of rule do you think governs this Nexiom machine?</p>
            <p style="margin: 10px 0 0 0; text-align: center; color: #6a1b9a; font-size: 0.9em;">Note: Since you are in the Q&A phase, you cannot interact with the objects or test the machine.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Conjunctive Rule**
            
            The machine switches on when **ALL** of the Nexioms are present on the machine.
            
            *Example: If Objects 1 and 3 are Nexioms, the machine only switches on when BOTH Object 1 AND Object 3 are on the machine.*
            """)
        
        with col2:
            st.markdown("""
            **Disjunctive Rule**
            
            The machine switches on when **ANY** of the Nexioms are present on the machine.
            
            *Example: If Objects 1 and 3 are Nexioms, the machine switches on when EITHER Object 1 OR Object 3 (or both) are on the machine.*
            """)
        
        rule_type = st.radio(
            "What type of rule do you think applies?",
            ["Conjunctive (ALL Nexioms must be present)", "Disjunctive (ANY Nexiom can activate)"],
            key="rule_type",
            index=None
        )
        
        # Navigation buttons
        st.markdown("---")
        st.markdown("### Submit Your Answers")
        
        # Check if rule type is provided
        rule_type = st.session_state.get("rule_type", "")
            
        # Get rule_hypothesis from saved_rule_hypothesis (saved when leaving text_area screen) or original widget key
        rule_hypothesis = st.session_state.get("saved_rule_hypothesis", "") or st.session_state.get("rule_hypothesis", "")
        print(f"🔍 Retrieved rule_hypothesis: '{rule_hypothesis[:50] if rule_hypothesis else 'EMPTY'}...'")
        print(f"🔍 Retrieved rule_type: '{rule_type}'")
        
        # Show Next Round button for all rounds except the last one
        if current_round + 1 < total_rounds:
            if st.button(" NEXT ROUND", type="primary", disabled=not rule_type, use_container_width=True):
                    # Get blicket classifications directly from saved tracked key
                    blicket_classifications = st.session_state.get("saved_blicket_classifications", {})
                    print(f"🔍 DEBUG: Using saved_blicket_classifications directly: {blicket_classifications}")
                
                    # Get rule hypothesis and rule type from session state
                    # Use saved_rule_hypothesis which was saved when leaving the text_area screen
                    saved_hypothesis = st.session_state.get("saved_rule_hypothesis", "")
                    widget_hypothesis = st.session_state.get("rule_hypothesis", "")
                    rule_hypothesis = saved_hypothesis if saved_hypothesis else widget_hypothesis
                    rule_type = st.session_state.get("rule_type", "")
                    
                    # Debug: Check all hypothesis sources
                    print(f"🔍 DEBUG: saved_rule_hypothesis = '{saved_hypothesis[:50] if saved_hypothesis else 'EMPTY'}...'")
                    print(f"🔍 DEBUG: widget rule_hypothesis = '{widget_hypothesis[:50] if widget_hypothesis else 'EMPTY'}...'")
                    print(f"🔍 DEBUG: final rule_hypothesis = '{rule_hypothesis[:50] if rule_hypothesis else 'EMPTY'}...'")
                    
                    # Debug: Print rule hypothesis and rule type
                    print(f"🔍 Round {current_round + 1}: rule_hypothesis = {rule_hypothesis[:100] if rule_hypothesis else 'EMPTY'}...")
                    print(f"🔍 Round {current_round + 1}: rule_type = '{rule_type}'")
                    print(f"🔍 Round {current_round + 1}: rule_type type = {type(rule_type)}")
                    print(f"🔍 Round {current_round + 1}: rule_type bool = {bool(rule_type)}")
                    print(f"🔍 Round {current_round + 1}: rule_type length = {len(rule_type) if rule_type else 0}")
                    
                    # Extract user's chosen blickets (objects marked as "Yes")
                    user_chosen_blickets = []
                    for i in range(round_config['num_objects']):
                        classification = blicket_classifications.get(f"object_{i}", "No")
                        print(f"🔍 Checking object_{i}: classification = '{classification}'")
                        if classification == "Yes":
                            user_chosen_blickets.append(i)  # 0-based index
                            print(f"   ✅ Added {i} to user_chosen_blickets")
                    
                    print(f"🔍 Final user_chosen_blickets: {user_chosen_blickets}")
                    
                    # Calculate total time spent on this round
                    end_time = datetime.datetime.now()
                    total_time_seconds = (end_time - st.session_state.game_start_time).total_seconds()
                    
                    # Generate unique round ID
                    now = datetime.datetime.now()
                    round_id = f"round_{current_round + 1}_{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
                    
                    # Create object labels mapping for this round
                    label_prefix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if is_practice else "1234567890"
                    object_labels_mapping = {}
                    for obj_idx in range(round_config['num_objects']):
                        label = label_prefix[obj_idx] if is_practice else str(obj_idx + 1)  # A/B/C or 1/2/3/4
                        object_labels_mapping[obj_idx] = label  # Map 0-based index to label
                    
                    # Map user_chosen_blickets to labels
                    user_chosen_blickets_with_labels = {}
                    for blicket_idx in user_chosen_blickets:
                        label = label_prefix[blicket_idx] if is_practice else str(blicket_idx + 1)
                        user_chosen_blickets_with_labels[label] = blicket_idx
                    
                    # Map blicket_classifications to labels
                    blicket_classifications_with_labels = {}
                    for obj_idx in range(round_config['num_objects']):
                        label = label_prefix[obj_idx] if is_practice else str(obj_idx + 1)
                        classification = blicket_classifications.get(f"object_{obj_idx}", "No")
                        blicket_classifications_with_labels[label] = {
                            "index": obj_idx,
                            "classification": classification
                        }
                    
                    # Save current round data with detailed action tracking
                    round_data = {
                        "round_id": round_id,  # Unique identifier for this round
                        "start_time": st.session_state.game_start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "total_time_seconds": total_time_seconds,
                        "round_number": current_round + 1,
                        "round_config": round_config,
                        "user_test_actions": st.session_state.user_test_actions,  # All place/remove/test actions with labels
                        "action_history": st.session_state.action_history,  # Detailed action history
                        "state_history": st.session_state.state_history,  # State changes with object labels
                        "test_timings": st.session_state.get("test_timings", []),  # Time for each test button press: [{"test_number": 1, "time_since_start_seconds": 2.5, "time_since_previous_seconds": 2.5}, ...]
                        "total_actions": len(st.session_state.user_test_actions),
                        "action_history_length": len(st.session_state.action_history),
                        "total_steps_taken": st.session_state.steps_taken,
                        "object_labels_mapping": object_labels_mapping,  # New: Mapping of indices to labels
                        "blicket_classifications": blicket_classifications,  # Objects picked as blickets (object_0, object_1, etc. with Yes/No answers)
                        "blicket_classifications_with_labels": blicket_classifications_with_labels,  # New: Classifications with labels
                        "user_chosen_blickets": sorted(user_chosen_blickets),  # User's chosen blicket indices [0, 2] (0-based)
                        "user_chosen_blickets_with_labels": user_chosen_blickets_with_labels,  # New: Chosen blickets with labels
                        "rule_hypothesis": rule_hypothesis,  # Rule inference input from user
                        "rule_type": rule_type if rule_type else "",  # Rule classification (conjunctive vs disjunctive)
                        "true_blicket_indices": convert_numpy_types(game_state['blicket_indices']),  # Ground truth blicket indices (0-based)
                        "true_rule": round_config['rule'],  # Ground truth rule for this round
                        "final_machine_state": bool(game_state['true_state'][-1]),
                        "final_objects_on_machine": list(st.session_state.selected_objects),
                        "rule": round_config['rule'],  # Keep for compatibility
                        "phase": "comprehension" if is_practice else "main_experiment",
                        "interface_type": "text"
                    }
                    
                    # Debug: Print what will be saved to Firebase
                    print(f"💾 Saving to Firebase - Round {current_round + 1}:")
                    print(f"   - rule_type in dict: '{round_data.get('rule_type', 'MISSING')}'")
                    print(f"   - rule_hypothesis in dict: '{round_data.get('rule_hypothesis', 'MISSING')[:50] if round_data.get('rule_hypothesis') else 'EMPTY'}...'")
                    print(f"   - user_chosen_blickets in dict: {round_data.get('user_chosen_blickets', 'MISSING')}")
                    print(f"   - blicket_classifications in dict: {round_data.get('blicket_classifications', 'MISSING')}")
                    
                    # Convert numpy types to ensure Firebase compatibility
                    round_data = convert_numpy_types(round_data)
                    print(f"💾 After convert_numpy_types - rule_type: '{round_data.get('rule_type', 'MISSING')}'")
                    
                    # Use the provided save function or default Firebase function
                    if save_data_func:
                        save_data_func(participant_id, round_data)
                    else:
                        save_game_data(participant_id, round_data)
                    
                    # Clean up old round_X_progress entries for main_game (if they exist)
                    if not is_practice:
                        try:
                            from firebase_admin import db
                            participant_ref = db.reference(f'{participant_id}')
                            main_game_ref = participant_ref.child('main_game')
                            progress_key = f"round_{current_round + 1}_progress"
                            if main_game_ref.child(progress_key).get():
                                main_game_ref.child(progress_key).delete()
                                print(f"🗑️ Deleted old {progress_key} entry")
                        except Exception as e:
                            print(f"⚠️ Could not delete old progress entry: {e}")
                    
                    # Clear session state for next round (but NOT the round counter or phase management)
                    # Only clear game-specific variables, not phase control variables
                    st.session_state.pop("visual_game_state", None)
                    st.session_state.pop("env", None)
                    st.session_state.pop("game_state", None)
                    st.session_state.pop("object_positions", None)
                    st.session_state.pop("shape_images", None)
                    st.session_state.pop("blicket_answers", None)
                    st.session_state.pop("game_start_time", None)
                    
                    # Clear Q&A variables
                    st.session_state.pop("rule_hypothesis", None)
                    st.session_state.pop("rule_type", None)
                    for i in range(10):
                        st.session_state.pop(f"blicket_q_{i}", None)
                    
                    # Note: Don't clear selected_objects, user_test_actions, action_history, state_history, steps_taken
                    # These will be reset by app.py's next_round handler
                    
                    # Return to main app for next round
                    st.session_state.phase = "next_round"
                    st.rerun()
            
            # Show message if button is disabled
            if not rule_type:
                st.markdown("<p style='color: #dc3545; font-size: 14px;'>Please select a rule type (Conjunctive or Disjunctive)</p>", unsafe_allow_html=True)
        else:
            # Show Finish Task button only on the last round
            # Check if rule type is provided
            rule_type = st.session_state.get("rule_type", "")
            
            if st.button("FINISH TASK", type="primary", disabled=not rule_type, use_container_width=True):
                    # Get blicket classifications directly from saved tracked key
                    blicket_classifications = st.session_state.get("saved_blicket_classifications", {})
                    print(f"🔍 DEBUG (FINAL): Using saved_blicket_classifications directly: {blicket_classifications}")
                
                    # Get rule hypothesis and rule type from session state
                    # Use saved_rule_hypothesis which was saved when leaving the text_area screen
                    rule_hypothesis = st.session_state.get("saved_rule_hypothesis", "") or st.session_state.get("rule_hypothesis", "")
                    rule_type = st.session_state.get("rule_type", "")
                    
                    # Debug: Print rule hypothesis
                    print(f"🔍 Round {current_round + 1} (FINAL): rule_hypothesis = {rule_hypothesis[:100] if rule_hypothesis else 'EMPTY'}...")
                    print(f"🔍 Round {current_round + 1} (FINAL): rule_type = {rule_type}")
                    
                    # Extract user's chosen blickets (objects marked as "Yes")
                    user_chosen_blickets = []
                    for i in range(round_config['num_objects']):
                        classification = blicket_classifications.get(f"object_{i}", "No")
                        print(f"🔍 Checking object_{i} (FINAL): classification = '{classification}'")
                        if classification == "Yes":
                            user_chosen_blickets.append(i)  # 0-based index
                            print(f"   ✅ Added {i} to user_chosen_blickets")
                    
                    print(f"🔍 Final user_chosen_blickets: {user_chosen_blickets}")
                    
                    # Get current game state from session state (ensure we have the latest)
                    current_game_state = st.session_state.get("game_state", game_state)
                    
                    # Calculate total time spent on this round
                    end_time = datetime.datetime.now()
                    total_time_seconds = (end_time - st.session_state.game_start_time).total_seconds()
                    
                    # Generate unique round ID for main phase
                    now = datetime.datetime.now()
                    round_id = f"main_round_{current_round + 1}_{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
                    
                    # Create object labels mapping for this round
                    label_prefix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if is_practice else "1234567890"
                    object_labels_mapping = {}
                    for obj_idx in range(round_config['num_objects']):
                        label = label_prefix[obj_idx] if is_practice else str(obj_idx + 1)  # A/B/C or 1/2/3/4
                        object_labels_mapping[obj_idx] = label  # Map 0-based index to label
                    
                    # Map user_chosen_blickets to labels
                    user_chosen_blickets_with_labels = {}
                    for blicket_idx in user_chosen_blickets:
                        label = label_prefix[blicket_idx] if is_practice else str(blicket_idx + 1)
                        user_chosen_blickets_with_labels[label] = blicket_idx
                    
                    # Map blicket_classifications to labels
                    blicket_classifications_with_labels = {}
                    for obj_idx in range(round_config['num_objects']):
                        label = label_prefix[obj_idx] if is_practice else str(obj_idx + 1)
                        classification = blicket_classifications.get(f"object_{obj_idx}", "No")
                        blicket_classifications_with_labels[label] = {
                            "index": obj_idx,
                            "classification": classification
                        }
                    
                    # Save final round data with detailed action tracking
                    round_data = {
                        "round_id": round_id,  # Unique identifier for this round
                        "start_time": st.session_state.game_start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "total_time_seconds": total_time_seconds,
                        "round_number": current_round + 1,
                        "round_config": round_config,
                        "user_test_actions": st.session_state.get("user_test_actions", []),  # All place/remove/test actions with labels
                        "action_history": st.session_state.get("action_history", []),  # Detailed action history
                        "state_history": st.session_state.get("state_history", []),  # State changes with object labels
                        "test_timings": st.session_state.get("test_timings", []),  # Time for each test button press
                        "total_actions": len(st.session_state.get("user_test_actions", [])),
                        "action_history_length": len(st.session_state.get("action_history", [])),
                        "total_steps_taken": st.session_state.get("steps_taken", 0),
                        "object_labels_mapping": object_labels_mapping,  # New: Mapping of indices to labels
                        "blicket_classifications": blicket_classifications,  # Objects picked as blickets (object_0, object_1, etc. with Yes/No answers)
                        "blicket_classifications_with_labels": blicket_classifications_with_labels,  # New: Classifications with labels
                        "user_chosen_blickets": sorted(user_chosen_blickets),  # User's chosen blicket indices [0, 2] (0-based)
                        "user_chosen_blickets_with_labels": user_chosen_blickets_with_labels,  # New: Chosen blickets with labels
                        "rule_hypothesis": rule_hypothesis,  # Rule inference input from user
                        "rule_type": rule_type if rule_type else "",  # Rule classification (conjunctive vs disjunctive)
                        "true_blicket_indices": convert_numpy_types(current_game_state.get('blicket_indices', round_config.get('blicket_indices', []))),  # Ground truth blicket indices (0-based)
                        "true_rule": round_config['rule'],  # Ground truth rule for this round
                        "final_machine_state": bool(current_game_state.get('true_state', [False])[-1]) if current_game_state else False,
                        "final_objects_on_machine": list(st.session_state.get("selected_objects", set())),
                        "rule": round_config['rule'],  # Keep for compatibility
                        "phase": "main_experiment",
                        "interface_type": "text"
                    }
                    
                    # Debug: Print what will be saved to Firebase
                    print(f"💾 Saving to Firebase - Round {current_round + 1} (FINAL):")
                    print(f"   - rule_type in dict: '{round_data.get('rule_type', 'MISSING')}'")
                    print(f"   - rule_hypothesis in dict: '{round_data.get('rule_hypothesis', 'MISSING')[:50] if round_data.get('rule_hypothesis') else 'EMPTY'}...'")
                    print(f"   - user_chosen_blickets in dict: {round_data.get('user_chosen_blickets', 'MISSING')}")
                    print(f"   - blicket_classifications in dict: {round_data.get('blicket_classifications', 'MISSING')}")
                    
                    # Convert numpy types to ensure Firebase compatibility
                    round_data = convert_numpy_types(round_data)
                    print(f"💾 After convert_numpy_types (FINAL) - rule_type: '{round_data.get('rule_type', 'MISSING')}'")
                    
                    # Use the provided save function or default Firebase function
                    if save_data_func:
                        save_data_func(participant_id, round_data)
                    else:
                        save_game_data(participant_id, round_data)
                    
                    # Clean up old round_X_progress entries for this round (if they exist)
                    try:
                        from firebase_admin import db
                        participant_ref = db.reference(f'{participant_id}')
                        main_game_ref = participant_ref.child('main_game')
                        progress_key = f"round_{current_round + 1}_progress"
                        if main_game_ref.child(progress_key).get():
                            main_game_ref.child(progress_key).delete()
                            print(f"🗑️ Deleted old {progress_key} entry (FINAL)")
                    except Exception as e:
                        print(f"⚠️ Could not delete old progress entry: {e}")
                    
                    # Clear session state completely for phase transition
                    reset_game_session_state()
                    
                    # Return to main app for completion
                    if is_practice:
                        st.session_state.phase = "practice_complete"
                    else:
                        st.session_state.phase = "end"
                    st.rerun()
            
            # Show message if button is disabled
            if not rule_type:
                st.markdown("<p style='color: #dc3545; font-size: 14px;'>Please select a rule type (Conjunctive or Disjunctive)</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Test the textual game
    test_config = {
        'num_objects': 4,
        'num_blickets': 2,
        'rule': 'conjunctive',
        'init_prob': 0.1,
        'transition_noise': 0.0
    }
    textual_blicket_game_page("test_user", test_config, 0, 3, None)
