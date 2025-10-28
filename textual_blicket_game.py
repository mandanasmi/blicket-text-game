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
        if phase == 'main_experiment':
            games_ref = participant_ref.child('main_game')
        else:
            games_ref = participant_ref.child('games')  # Keep comprehension phase data in 'games'
        
        # Create a new game entry with detailed timestamp
        now = datetime.datetime.now()
        game_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Enhance game_data with additional metadata
        enhanced_game_data = {
            **game_data,
            "saved_at": now.isoformat(),
            "game_id": game_id,
            "session_timestamp": now.timestamp()
        }
        
        games_ref.child(game_id).set(enhanced_game_data)
        print(f"✅ Successfully saved {phase} data for {participant_id} - Game ID: {game_id}")
    except Exception as e:
        print(f"❌ Failed to save game data for {participant_id}: {e}")
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
        qa_data = {
            "round_id": round_id,
            "start_time": st.session_state.game_start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time_seconds": total_time_seconds,
            "round_number": current_round + 1,
            "round_config": round_config,
            "user_actions": st.session_state.user_actions.copy() if 'user_actions' in st.session_state else [],
            "action_history": st.session_state.action_history.copy() if 'action_history' in st.session_state else [],
            "state_history": st.session_state.state_history.copy() if 'state_history' in st.session_state else [],
            "total_actions": len(st.session_state.user_actions) if 'user_actions' in st.session_state else 0,
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
        "shape_images", "steps_taken", "user_actions", "action_history", 
        "state_history", "rule_hypothesis", "rule_type"
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
        # Determine which key to use based on phase
        phase = "comprehension" if is_practice else "main_experiment"
        
        # Check if Firebase is initialized
        if not firebase_admin._apps:
            print(f"⚠️ Firebase not initialized - skipping {phase} progress save")
            return
        
        # Get database reference
        db_ref = db.reference()
        participant_ref = db_ref.child(participant_id)
        
        if phase == "main_experiment":
            progress_ref = participant_ref.child('main_game')
            entry_id = f"round_{current_round + 1}_progress"
        else:
            progress_ref = participant_ref.child('comprehension')
            entry_id = "action_history"
        
        # Get existing data or create new
        existing_data = progress_ref.child(entry_id).get() or {}
        
        # Update with current action history and Q&A data
        # Note: We don't save round_config here - it's already in the final round data
        now = datetime.datetime.now()
        updated_data = {
            **existing_data,
            "last_updated": now.isoformat(),
            "user_actions": st.session_state.user_actions.copy() if 'user_actions' in st.session_state else [],
            "total_actions": len(st.session_state.user_actions) if 'user_actions' in st.session_state else 0,
            "action_history": st.session_state.action_history.copy() if 'action_history' in st.session_state else [],
            "total_steps_taken": st.session_state.steps_taken if 'steps_taken' in st.session_state else 0,
            "selected_objects": list(st.session_state.selected_objects) if 'selected_objects' in st.session_state else [],
            "game_start_time": st.session_state.game_start_time.isoformat() if 'game_start_time' in st.session_state else now.isoformat(),
            "phase": phase,
            "round_number": current_round + 1
        }
        
        # Add Q&A data if provided
        if blicket_classifications is not None:
            updated_data["blicket_classifications"] = blicket_classifications
        if rule_hypothesis is not None:
            updated_data["rule_hypothesis"] = rule_hypothesis
        if rule_type is not None:
            updated_data["rule_type"] = rule_type
        if objects_on_machine is not None:
            updated_data["objects_on_machine_before_qa"] = objects_on_machine
        
        # Convert NumPy types to JSON-serializable types
        updated_data = convert_numpy_types(updated_data)
        
        progress_ref.child(entry_id).set(updated_data)
        print(f"💾 {phase} progress updated for {participant_id} - Round {current_round + 1} - Q&A data included")
        
    except Exception as e:
        import traceback
        print(f"❌ Failed to save {phase} progress for {participant_id}: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

def textual_blicket_game_page(participant_id, round_config, current_round, total_rounds, save_data_func=None, use_visual_mode=None, is_practice=False):
    """Main blicket game page - text-only interface"""
    
    # Determine which mode to use - this is the LOCAL variable for this function
    if use_visual_mode is not None:
        use_text_version = not use_visual_mode
    else:
        # Fall back to global setting or environment variable
        use_text_version = os.getenv('BLICKET_VISUAL_MODE', 'False').lower() != 'true'
    
    # Simple CSS for clean button styling
    st.markdown("""
    <style>
    /* Clean button styling - only green/gray colors */
    .stApp .stButton button[kind="primary"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin: 5px !important;
        font-weight: bold !important;
    }
    
    .stApp .stButton button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    
    .stApp .stButton button[kind="secondary"] {
        background-color: #6c757d !important;
        border-color: #6c757d !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin: 5px !important;
        font-weight: bold !important;
    }
    
    .stApp .stButton button[kind="secondary"]:hover {
        background-color: #5a6268 !important;
        border-color: #545b62 !important;
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
        st.session_state.user_actions = []  # Track all user actions for Firebase
        st.session_state.action_history = []  # Track action history for text version
        st.session_state.state_history = []  # Track complete state history
        
        print(f"🔄 New round initialized - Round {current_round + 1}/{total_rounds}")
        print(f"   - Objects: {round_config['num_objects']}")
        print(f"   - Blickets: {round_config['num_blickets']}")
        print(f"   - Rule: {round_config['rule']}")
        print(f"   - Blicket indices: {round_config.get('blicket_indices', 'None')}")
        
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
    
    # Create sidebar for state history
    with st.sidebar:
        st.markdown("### State History")
        
        # Create a container with fixed height and scrollbar
        history_container = st.container()
        
        with history_container:
            if st.session_state.state_history:
                for i, state in enumerate(st.session_state.state_history):
                    if use_text_version:
                        # Text version: show object numbers with green highlighting
                        objects_text = ""
                        for obj_idx in range(round_config['num_objects']):
                            object_id = obj_idx  # 0-based object ID
                            if object_id in state['objects_on_machine']:
                                objects_text += f"<span style='background-color: #00ff00; color: black; padding: 1px 4px; margin: 1px; border-radius: 2px; font-size: 12px;'>{object_id}</span>"
                            else:
                                objects_text += f"<span style='background-color: #333333; color: white; padding: 1px 4px; margin: 1px; border-radius: 2px; font-size: 12px;'>{object_id}</span>"
                        
                        # Show machine state on same row
                        machine_status = "🟢" if state['machine_lit'] else "🔴"
                        st.markdown(f"<div style='margin: 2px 0; font-size: 12px;'><strong>{i + 1}:</strong> {objects_text} {machine_status}</div>", unsafe_allow_html=True)
                    else:
                        # Visual version: show object numbers with colored backgrounds (much faster than images)
                        cols = st.columns(round_config['num_objects'] + 2)  # +2 for step number and machine status
                        
                        with cols[0]:
                            st.markdown(f"<div style='font-size: 12px; margin: 2px 0;'><strong>{i + 1}:</strong></div>", unsafe_allow_html=True)
                        
                        # Show each object
                        for obj_idx in range(round_config['num_objects']):
                            object_id = obj_idx  # 0-based object ID
                            with cols[obj_idx + 1]:
                                if object_id in state['objects_on_machine']:
                                    # Selected object - green background
                                    st.markdown(f"""
                                    <div style="
                                        background-color: #00ff00; 
                                        border: 1px solid #00ff00; 
                                        border-radius: 3px; 
                                        padding: 2px 4px; 
                                        margin: 1px; 
                                        text-align: center;
                                        color: black;
                                        font-size: 10px;
                                        font-weight: bold;
                                    ">
                                        {obj_idx + 1}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # Unselected object - gray background
                                    st.markdown(f"""
                                    <div style="
                                        background-color: #333333; 
                                        border: 1px solid #666666; 
                                        border-radius: 3px; 
                                        padding: 2px 4px; 
                                        margin: 1px; 
                                        text-align: center;
                                        color: white;
                                        font-size: 10px;
                                        font-weight: bold;
                                    ">
                                        {obj_idx + 1}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Show machine status
                        with cols[-1]:
                            machine_status = "🟢" if state['machine_lit'] else "🔴"
                            st.markdown(f"<div style='font-size: 12px; margin: 2px 0;'>{machine_status}</div>", unsafe_allow_html=True)
            else:
                st.markdown("*No states recorded yet.*")
    
    # Main content area
    # Display round info and progress
    st.markdown(f"## Round {current_round + 1} of {total_rounds}")
    
    # Progress bar
    progress = (current_round + 1) / total_rounds
    st.progress(progress)
    
    # Collapsible instruction section
    with st.expander("📋 Game Instructions", expanded=False):
        horizon = round_config.get('horizon', 32)  # Default to 32 steps
        st.markdown(f"""

        **Your goals are:**
        - Identify which objects are blickets.
        - Infer the underlying rule for how the machine turns on. 

        **Tips:**
        - All objects can be either on the machine or on the floor.
        - You should think about how to efficiently explore the relationship between the objects and the machine.

        You have **{horizon} steps** to complete the task. You can also exit the task early if you think you understand the relationship between the objects and the machine. After the task is done, you will be asked which objects are blickets, and the rule for turning on the machine.

        You will be prompted at each turn to choose actions.
        """)
    

    
    # Text-only version: Display action history
    if use_text_version:
        st.markdown("### Action History")
        if st.session_state.action_history:
            for action_text in st.session_state.action_history:
                st.markdown(f"• {action_text}")
        else:
            st.markdown("*No actions taken yet.*")
        st.markdown("---")
    
    # Display the blicket machine (only in visual version)
    if not use_text_version:
        st.markdown("### The Blicket Machine")
    
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
                    <div style="margin-top: 10px; font-size: 18px; font-weight: bold; color: {'#00ff00' if machine_lit else '#ff4444'};">
                        Blicket Detector {'LIT' if machine_lit else 'NOT LIT'}
                    </div>
                </div>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px 25px; border-radius: 15px; color: white; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                    <div style="font-size: 14px; margin-bottom: 5px;">Steps Remaining</div>
                    <div style="font-size: 24px;">{steps_left}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Text-only version: Show steps counter and machine status
    if use_text_version:
        horizon = round_config.get('horizon', 32)
        steps_left = horizon - st.session_state.steps_taken
        machine_status = "🟢 LIT" if machine_lit else "🔴 NOT LIT"
        
        st.markdown(f"""
        ### 🏭 Blicket Detector Status: {machine_status}
        **Steps Remaining: {steps_left}/{horizon}**
        """)
        st.markdown("---")
    
    # Show warning if no steps left
    if steps_left <= 0:
        st.markdown("""
        <div style="background: rgba(220, 53, 69, 0.8); border: 1px solid rgba(220, 53, 69, 0.9); border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center;">
            <strong>No steps remaining!</strong>
        </div>
        """, unsafe_allow_html=True)
    

    
    # Display available objects
    st.markdown("### Available Objects")
    
    # Button styling is handled at the top of the function
    
    if use_text_version:
        st.markdown("""
        **How to use the interface:**
        - Click on an object button to **place** it on the blicket detector
        - The button will turn **green** when the object is placed
        - Click the button again to **remove** the object from the detector
        - The button will turn **gray** when the object is removed
        - Each button shows the current state: **green = placed**, **gray = not placed**
        - Try different combinations to figure out which objects are blickets!
        """)
        
        # Text-only version: Simple button grid
        cols = st.columns(4)
        for i in range(round_config['num_objects']):
            with cols[i % 4]:
                object_id = i  # 0-based object ID
                is_selected = object_id in st.session_state.selected_objects
                horizon = round_config.get('horizon', 32)
                steps_left = horizon - st.session_state.steps_taken
                interaction_disabled = (steps_left <= 0 or st.session_state.visual_game_state == "questionnaire")
                
                # Single button that toggles between green (selected) and gray (unselected)
                button_type = "primary" if is_selected else "secondary"
                
                if st.button(f"Object {i}", 
                           key=f"obj_{i}", 
                           disabled=interaction_disabled,
                           type=button_type,
                           help=f"Click to {'remove' if is_selected else 'place'} Object {i}"):
                    # Record the action before making changes
                    action_time = datetime.datetime.now()
                    action_type = "remove" if is_selected else "place"
                    
                    # Capture machine state BEFORE making changes
                    machine_state_before = bool(game_state['true_state'][-1]) if 'game_state' in st.session_state else False
                    
                    # Update object selection
                    if is_selected:
                        st.session_state.selected_objects.remove(object_id)
                    else:
                        st.session_state.selected_objects.add(object_id)
                    
                    # Update environment state (convert back to 0-based for internal state)
                    env._state[i] = (object_id in st.session_state.selected_objects)
                    env._update_machine_state()
                    game_state = env.step("look")[0]  # Get updated state
                    st.session_state.game_state = game_state
                    
                    # Add to action history
                    action_text = f"You {'removed' if action_type == 'remove' else 'put'} Object {object_id} {'from' if action_type == 'remove' else 'on'} the machine. The blicket detector is {'LIT' if game_state['true_state'][-1] else 'NOT LIT'}."
                    st.session_state.action_history.append(action_text)
                    
                    # Add to state history
                    state_data = {
                        "objects_on_machine": set(st.session_state.selected_objects),
                        "machine_lit": bool(game_state['true_state'][-1]),
                        "step_number": st.session_state.steps_taken + 1
                    }
                    st.session_state.state_history.append(state_data)
                    
                    # Record the action for Firebase
                    action_data = {
                        "timestamp": action_time.isoformat(),
                        "action_type": action_type,
                        "object_index": object_id,
                        "object_id": f"object_{object_id}",
                        "machine_state_before": machine_state_before,  # Machine state before this action
                        "machine_state_after": bool(game_state['true_state'][-1]),  # New state
                        "objects_on_machine": list(st.session_state.selected_objects),
                        "step_number": st.session_state.steps_taken + 1
                    }
                    st.session_state.user_actions.append(action_data)
                    
                    # Increment step counter
                    st.session_state.steps_taken += 1
                    
                    # Save intermediate progress after each action (only for comprehension phase)
                    if is_practice:
                        save_intermediate_progress(participant_id, round_config, current_round, total_rounds, is_practice)
                    
                    st.rerun()
    else:
        st.markdown("Click on an object to place it on the machine. Click again to remove it.")
        
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
                    interaction_disabled = (steps_left <= 0 or st.session_state.visual_game_state == "questionnaire")
                    
                    # Create clickable image with improved styling
                    if interaction_disabled:
                        # Disabled state - gray out the image
                        opacity = "0.5"
                        cursor = "not-allowed"
                        border_color = "#cccccc"
                    else:
                        opacity = "1.0"
                        cursor = "pointer"
                        border_color = "#00ff00" if is_selected else "#ffffff"
                    
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
                                background: {'rgba(0, 255, 0, 0.2)' if is_selected else 'transparent'};
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                                opacity: {opacity};
                            ">
                                <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 80px; height: auto; margin-bottom: 10px;">
                                <br>
                                <div style="font-weight: bold; color: {'#00ff00' if is_selected else '#333'}; font-size: 16px;">
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
                                background: {'rgba(0, 255, 0)' if is_selected else 'transparent'};
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                                transition: all 0.2s ease;
                                position: relative;
                            " onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 6px 20px rgba(0,0,0,0.15)'" onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.1)'">
                                <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 80px; height: auto; margin-bottom: 10px;">
                                <br>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Select Object {obj_idx}", key=f"obj_{obj_idx}", help=f"Click to {'remove' if is_selected else 'place'} Object {obj_idx}"):
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
                            state_data = {
                                "objects_on_machine": set(st.session_state.selected_objects),
                                "machine_lit": bool(game_state['true_state'][-1]),
                                "step_number": st.session_state.steps_taken + 1
                            }
                            st.session_state.state_history.append(state_data)
                            
                            # Record the action for Firebase
                            action_data = {
                                "timestamp": action_time.isoformat(),
                                "action_type": action_type,
                                "object_index": obj_idx - 1,
                                "object_id": f"object_{obj_idx}",
                                "machine_state_before": machine_state_before,  # Machine state before this action
                                "machine_state_after": bool(game_state['true_state'][-1]),  # New state
                                "objects_on_machine": list(st.session_state.selected_objects),
                                "step_number": st.session_state.steps_taken + 1
                            }
                            st.session_state.user_actions.append(action_data)
                            
                            # Increment step counter
                            st.session_state.steps_taken += 1
                            
                            # Save intermediate progress after each action (only for comprehension phase)
                            if is_practice:
                                save_intermediate_progress(participant_id, round_config, current_round, total_rounds, is_practice)
                            
                            st.rerun()
                    
                    
    

    
    # Phase transition buttons
    st.markdown("---")
    st.markdown("### 🎯 Game Controls")
    
    if st.session_state.visual_game_state == "exploration":
        horizon = round_config.get('horizon', 32)
        steps_left = horizon - st.session_state.steps_taken
        
        # Show different buttons based on steps remaining
        if steps_left <= 0:
            if is_practice:
                # Comprehension phase - no questions, just show completion message
                st.markdown("""
                <div style="background: rgba(40, 167, 69, 0.8); border: 2px solid #28a745; border-radius: 10px; padding: 20px; margin: 20px 0; text-align: center;">
                    <h3 style="color: #155724; margin: 0;">⏰ No Remaining Steps</h3>
                    <p style="color: #155724; margin: 10px 0;">Now that the comprehension phase is over, you can play the main game!</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Start Main Game", type="primary", key="complete_btn", use_container_width=True):
                    # Return to main app for completion
                    st.session_state.phase = "practice_complete"
                    st.rerun()
            else:
                # Main experiment - show questionnaire
                st.markdown("""
                <div style="background: rgba(255, 193, 7, 0.8); border: 2px solid #ffc107; border-radius: 10px; padding: 20px; margin: 20px 0; text-align: center;">
                    <h3 style="color: #856404; margin: 0;">⏰ No Steps Remaining!</h3>
                    <p style="color: #856404; margin: 10px 0;">Please proceed to answer questions about which objects are blickets.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 PROCEED TO ANSWER QUESTIONS", type="primary", key="proceed_btn", use_container_width=True):
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
            st.markdown(f"""
            <div style="background: rgba(13, 202, 240, 0.1); border: 2px solid #0dcaf0; border-radius: 10px; padding: 20px; margin: 20px 0; text-align: center;">
                <h3 style="color: #0dcaf0; margin: 0;">📊 You have {steps_left} steps remaining</h3>
                <p style="color: #0dcaf0; margin: 10px 0;">You can continue exploring or proceed to answer questions about blickets.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show different button based on phase
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if is_practice:
                    # Comprehension phase - no questions, just show completion option
                    if st.button("🚀 Start Main Game", type="primary", key="complete_ready_btn", use_container_width=True):
                        # Return to main app for completion
                        st.session_state.phase = "practice_complete"
                        st.rerun()
                else:
                    # Main experiment - show questionnaire button
                    if st.button("🎯 READY TO ANSWER QUESTIONS", type="primary", key="ready_btn", use_container_width=True):
                        # Clear any previous blicket answers to ensure fresh start
                        for i in range(round_config['num_objects']):
                            if f"blicket_q_{i}" in st.session_state:
                                del st.session_state[f"blicket_q_{i}"]
                        st.session_state.visual_game_state = "questionnaire"
                        st.rerun()
            
            st.markdown(f"**Steps remaining: {steps_left}/{horizon}**")
    
    elif st.session_state.visual_game_state == "questionnaire" and not is_practice:
        st.markdown("""
        <div style="padding: 20px; border-radius: 15px; background-color: #e3f2fd; border: 2px solid #2196f3; margin: 20px 0;">
            <h3 style="margin: 0; text-align: center; color: #1565c0;">🎯 Blicket Classification</h3>
            <p style="margin: 10px 0 0 0; text-align: center; color: #1976d2;">For each object, indicate whether you think it is a blicket or not:</p>
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
                st.markdown(f"**Object {i}**")
                st.radio(
                    f"Is Object {i} a blicket?",
                    ["Yes", "No"],
                    key=f"blicket_q_{i}",
                    index=None
                )
                st.markdown("---")
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
                    f"Is Object {i + 1} a blicket?",
                    ["Yes", "No"],
                    key=f"blicket_q_{i}",
                    index=None
                )
                
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Add rule question
        st.markdown("---")
        st.markdown("### Rule Inference")
        st.markdown("Based on your observations, what do you think is the rule for how the blicket detector works?")
        st.text_area(
            "What do you think is the rule?",
            placeholder="Describe your hypothesis about how the blicket detector determines when to light up...",
            height=100,
            key="rule_hypothesis"
        )
        
        # Navigation buttons - allow users to save and continue or finish
        st.markdown("---")
        st.markdown("### 🚀 Continue to Rule Type Classification")
        
        # Check if rule hypothesis is provided (read from session state)
        rule_hypothesis = st.session_state.get("rule_hypothesis", "")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Always show the button, but check inside the callback
            if st.button("➡️ NEXT: Rule Type Classification", type="primary", use_container_width=True):
                # Check if rule hypothesis is provided
                current_hypothesis = st.session_state.get("rule_hypothesis", "").strip()
                print(f"🔍 DEBUG: Raw rule_hypothesis from widget: '{st.session_state.get('rule_hypothesis', 'NOT FOUND')}'")
                print(f"🔍 DEBUG: Trimmed current_hypothesis: '{current_hypothesis}'")
                
                if current_hypothesis:
                    # Save blicket classifications before moving to rule type
                    blicket_classifications = {}
                    print(f"🔍 DEBUG: About to collect blicket answers, num_objects = {round_config['num_objects']}")
                    for i in range(round_config['num_objects']):
                        raw_answer = st.session_state.get(f"blicket_q_{i}", None)
                        print(f"🔍 DEBUG: Raw blicket_q_{i} from session state: {raw_answer}")
                        answer = raw_answer if raw_answer is not None else "No"
                        blicket_classifications[f"object_{i}"] = answer
                        print(f"🔍 Saving intermediate - blicket_q_{i} = {answer}")
                    
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
                        print(f"✅ Saved intermediate progress for comprehension phase")
                    
                    print(f"📝 Preparing to save hypothesis for round {current_round + 1}")
                    print(f"   - Objects on machine: {objects_on_machine_before_qa}")
                    print(f"   - Blicket classifications: {blicket_classifications}")
                    print(f"   - Hypothesis: {current_hypothesis[:50]}...")
                    
                    # Debug: Check session state before transition
                    print(f"🔍 DEBUG: Before transitioning to rule_type, checking session state:")
                    print(f"   - rule_hypothesis: {st.session_state.get('rule_hypothesis', 'NOT FOUND')}")
                    for i in range(round_config['num_objects']):
                        key = f"blicket_q_{i}"
                        if key in st.session_state:
                            print(f"   - {key}: {st.session_state.get(key, 'NOT FOUND')}")
                    
                    # Keep blicket answers in session state for the rule type classification phase
                    for i in range(round_config['num_objects']):
                        if f"blicket_q_{i}" not in st.session_state:
                            st.session_state[f"blicket_q_{i}"] = blicket_classifications.get(f"object_{i}", "No")
                    
                    # Preserve hypothesis in a separate key that won't be cleared by widget lifecycle
                    # The widget key "rule_hypothesis" won't persist once we leave this screen
                    st.session_state["saved_rule_hypothesis"] = current_hypothesis
                    print(f"🔍 DEBUG: Saved rule_hypothesis to saved_rule_hypothesis key: {current_hypothesis[:50]}...")
                    
                    st.session_state.visual_game_state = "rule_type_classification"
                    st.rerun()
                else:
                    st.warning("Please enter a rule hypothesis before proceeding.")

    elif st.session_state.visual_game_state == "rule_type_classification" and not is_practice:
        st.markdown("""
        <div style="padding: 20px; border-radius: 15px; background-color: #f3e5f5; border: 2px solid #9c27b0; margin: 20px 0;">
            <h3 style="margin: 0; text-align: center; color: #4a148c;">🎯 Rule Type Classification</h3>
            <p style="margin: 10px 0 0 0; text-align: center; color: #6a1b9a;">Based on your observations, what type of rule do you think governs the blicket detector?</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔄 Conjunctive Rule**
            
            The machine lights up when **ALL** of the blickets are present on the machine.
            
            *Example: If Objects 1 and 3 are blickets, the machine only lights up when BOTH Object 1 AND Object 3 are on the machine.*
            """)
        
        with col2:
            st.markdown("""
            **🔀 Disjunctive Rule**
            
            The machine lights up when **ANY** of the blickets are present on the machine.
            
            *Example: If Objects 1 and 3 are blickets, the machine lights up when EITHER Object 1 OR Object 3 (or both) are on the machine.*
            """)
        
        rule_type = st.radio(
            "What type of rule do you think applies?",
            ["Conjunctive (ALL blickets must be present)", "Disjunctive (ANY blicket can activate)"],
            key="rule_type",
            index=None
        )
        
        # Navigation buttons
        st.markdown("---")
        st.markdown("### 🚀 Submit Your Answers")
        
        # Check if rule type is provided
        rule_type = st.session_state.get("rule_type", "")
        
        # Get rule_hypothesis from saved_rule_hypothesis (saved when leaving text_area screen) or original widget key
        rule_hypothesis = st.session_state.get("saved_rule_hypothesis", "") or st.session_state.get("rule_hypothesis", "")
        print(f"🔍 Retrieved rule_hypothesis: '{rule_hypothesis[:50] if rule_hypothesis else 'EMPTY'}...'")
        print(f"🔍 Retrieved rule_type: '{rule_type}'")
        
        # Show Next Round button for all rounds except the last one
        if current_round + 1 < total_rounds:
            col1, col2, col3 = st.columns([1, 2, 1])
            with             col2:
                if st.button("➡️ NEXT ROUND", type="primary", disabled=not rule_type, use_container_width=True):
                    # Collect blicket classifications (using 0-based object IDs)
                    blicket_classifications = {}
                    
                    # Debug: Check all blicket_q keys in session state
                    print(f"🔍 DEBUG: Checking session state for blicket answers...")
                    print(f"🔍 DEBUG: All keys in session state: {list(st.session_state.keys())}")
                    for key in sorted(st.session_state.keys()):
                        if key.startswith("blicket_q_"):
                            print(f"   ✅ Found key: {key} = '{st.session_state.get(key)}'")
                    for i in range(round_config['num_objects']):
                        key = f"blicket_q_{i}"
                        raw_value = st.session_state.get(key, "MISSING_KEY")
                        print(f"🔍 DEBUG: blicket_q_{i}: raw_value = '{raw_value}'")
                        answer = "No" if raw_value == "MISSING_KEY" else raw_value
                        blicket_classifications[f"object_{i}"] = answer
                        print(f"🔍 Round {current_round + 1}: blicket_q_{i} = {answer}")
                
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
                    
                    # Save current round data with detailed action tracking
                    round_data = {
                        "round_id": round_id,  # Unique identifier for this round
                        "start_time": st.session_state.game_start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "total_time_seconds": total_time_seconds,
                        "round_number": current_round + 1,
                        "round_config": round_config,
                        "user_actions": st.session_state.user_actions,  # All place/remove actions
                        "action_history": st.session_state.action_history,  # Detailed action history
                        "state_history": st.session_state.state_history,  # State changes
                        "total_actions": len(st.session_state.user_actions),
                        "action_history_length": len(st.session_state.action_history),  # Length of action history
                        "total_steps_taken": st.session_state.steps_taken,
                        "blicket_classifications": blicket_classifications,  # Objects picked as blickets during question-answering (object_0, object_1, etc. with Yes/No answers)
                        "user_chosen_blickets": sorted(user_chosen_blickets),  # User's chosen blicket indices [0, 2] for objects 0 and 2 marked as Yes
                        "rule_hypothesis": rule_hypothesis,  # Hypothesis written in the text box
                        "rule_type": rule_type if rule_type else "",  # Hypothesis chosen in the last question (conjunctive vs disjunctive) - FORCE string
                        "true_blicket_indices": convert_numpy_types(game_state['blicket_indices']),  # Ground truth blicket indices
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
                    
                    # Note: Don't clear selected_objects, user_actions, action_history, state_history, steps_taken
                    # These will be reset by app.py's next_round handler
                    
                    # Return to main app for next round
                    st.session_state.phase = "next_round"
                    st.rerun()
        else:
            # Show Finish Task button only on the last round
            # Check if rule type is provided
            rule_type = st.session_state.get("rule_type", "")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🏁 FINISH TASK", type="primary", disabled=not rule_type, use_container_width=True):
                    # Collect blicket classifications (using 0-based object IDs)
                    blicket_classifications = {}
                    for i in range(round_config['num_objects']):
                        answer = st.session_state.get(f"blicket_q_{i}", "No")
                        blicket_classifications[f"object_{i}"] = answer
                        print(f"🔍 Round {current_round + 1} (FINAL): blicket_q_{i} = {answer}")
                
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
                    
                    # Save final round data with detailed action tracking
                    round_data = {
                        "round_id": round_id,  # Unique identifier for this round
                        "start_time": st.session_state.game_start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "total_time_seconds": total_time_seconds,
                        "round_number": current_round + 1,
                        "round_config": round_config,
                        "user_actions": st.session_state.get("user_actions", []),  # All place/remove actions
                        "action_history": st.session_state.get("action_history", []),  # Detailed action history
                        "state_history": st.session_state.get("state_history", []),  # State changes
                        "total_actions": len(st.session_state.get("user_actions", [])),
                        "action_history_length": len(st.session_state.get("action_history", [])),  # Length of action history
                        "total_steps_taken": st.session_state.get("steps_taken", 0),
                        "blicket_classifications": blicket_classifications,  # Objects picked as blickets during question-answering (object_0, object_1, etc. with Yes/No answers)
                        "user_chosen_blickets": sorted(user_chosen_blickets),  # User's chosen blicket indices [0, 2] for objects 0 and 2 marked as Yes
                        "rule_hypothesis": rule_hypothesis,  # Hypothesis written in the text box
                        "rule_type": rule_type if rule_type else "",  # Hypothesis chosen in the last question (conjunctive vs disjunctive) - FORCE string
                        "true_blicket_indices": convert_numpy_types(current_game_state.get('blicket_indices', round_config.get('blicket_indices', []))),  # Ground truth blicket indices
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
                    
                    # Clear session state completely for phase transition
                    reset_game_session_state()
                    
                    # Return to main app for completion
                    if is_practice:
                        st.session_state.phase = "practice_complete"
                    else:
                        st.session_state.phase = "end"
                    st.rerun()

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
