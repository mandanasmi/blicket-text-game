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
USE_TEXT_VERSION = False

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
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

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
    return env, game_state

def save_game_data(participant_id, game_data):
    """Save game data to Firebase"""
    # Convert NumPy types to JSON-serializable types
    game_data = convert_numpy_types(game_data)
    
    # Get database reference
    db_ref = db.reference()
    participant_ref = db_ref.child(participant_id)
    games_ref = participant_ref.child('games')
    
    # Create a new game entry with timestamp
    game_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    games_ref.child(game_id).set(game_data)
    
    print(f"Saving game data for {participant_id}: {game_data}")

def visual_blicket_game_page(participant_id, round_config, current_round, total_rounds, save_data_func=None):
    """Main visual blicket game page"""
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .object-highlight {
        border: 2px solid #00ff00;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    .object-normal {
        border: 2px solid #cccccc;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    
    /* Style the toggle buttons to be more compact */
    div[data-testid="stButton"] > button {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin: 5px 0 !important;
        color: white !important;
        font-size: 12px !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stButton"] > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for this page
    if "visual_game_state" not in st.session_state:
        st.session_state.visual_game_state = "exploration"
        st.session_state.env, st.session_state.game_state = create_new_game(
            seed=42 + current_round,
            num_objects=round_config['num_objects'],
            num_blickets=round_config['num_blickets'],
            rule=round_config['rule']
        )
        st.session_state.object_positions = {}  # Track object positions
        st.session_state.selected_objects = set()  # Objects currently on machine
        st.session_state.blicket_answers = {}  # User's blicket classifications
        st.session_state.game_start_time = datetime.datetime.now()
        st.session_state.steps_taken = 0  # Track number of steps taken
        st.session_state.user_actions = []  # Track all user actions for Firebase
        st.session_state.action_history = []  # Track action history for text version
        st.session_state.state_history = []  # Track complete state history
        
        # Initialize fixed shape images for this round (ensure different images)
        if not USE_TEXT_VERSION:
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
    
    # Use fixed shape images from session state (only for visual version)
    if not USE_TEXT_VERSION:
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
    
    # Create sidebar for state history
    with st.sidebar:
        st.markdown("### State History")
        
        # Create a container with fixed height and scrollbar
        history_container = st.container()
        
        with history_container:
            if st.session_state.state_history:
                for i, state in enumerate(st.session_state.state_history):
                    if USE_TEXT_VERSION:
                        # Text version: show object numbers with green highlighting
                        objects_text = ""
                        for obj_idx in range(round_config['num_objects']):
                            if obj_idx in state['objects_on_machine']:
                                objects_text += f"<span style='background-color: #00ff00; color: black; padding: 1px 4px; margin: 1px; border-radius: 2px; font-size: 12px;'>{obj_idx + 1}</span>"
                            else:
                                objects_text += f"<span style='background-color: #333333; color: white; padding: 1px 4px; margin: 1px; border-radius: 2px; font-size: 12px;'>{obj_idx + 1}</span>"
                        
                        # Show machine state on same row
                        machine_status = "🟢" if state['machine_lit'] else "🔴"
                        st.markdown(f"<div style='margin: 2px 0; font-size: 12px;'><strong>{i + 1}:</strong> {objects_text} {machine_status}</div>", unsafe_allow_html=True)
                    else:
                        # Visual version: show object images with green highlighting
                        # Create columns for compact display
                        cols = st.columns(round_config['num_objects'] + 2)  # +2 for step number and machine status
                        
                        with cols[0]:
                            st.markdown(f"<div style='font-size: 12px; margin: 2px 0;'><strong>{i + 1}:</strong></div>", unsafe_allow_html=True)
                        
                        # Show each object
                        for obj_idx in range(round_config['num_objects']):
                            with cols[obj_idx + 1]:
                                if obj_idx in state['objects_on_machine']:
                                    # Selected object - green background
                                    st.markdown(f"""
                                    <div style="
                                        background-color: rgba(0, 255, 0, 0.3); 
                                        border: 1px solid #00ff00; 
                                        border-radius: 3px; 
                                        padding: 2px; 
                                        margin: 1px; 
                                        text-align: center;
                                    ">
                                        <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 20px; height: auto;">
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # Unselected object - transparent background
                                    st.markdown(f"""
                                    <div style="
                                        background-color: transparent; 
                                        border: 1px solid #cccccc; 
                                        border-radius: 3px; 
                                        padding: 2px; 
                                        margin: 1px; 
                                        text-align: center;
                                    ">
                                        <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 20px; height: auto;">
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
        You are an intelligent, curious agent. You are playing a game where you are in a room with different objects, and a machine. Some of these objects are blickets. You can't tell which object is a blicket just by looking at it, but they have blicktness inside of them. Blicktness makes the machine turn on, following some hidden rule.

        **Your goals are:**
        - Identify which objects are blickets.
        - Infer the underlying rule for how the machine turns on. 

        **Here are the available commands:**
        - **look:** describe the current room
        - **put ... on ...:** put an object on the machine or the floor
        - **take ... off ...:** take an object off the machine
        - **exit:** exit the game

        **Tips:**
        - All objects can be either on the machine or on the floor.
        - You should think about how to efficiently explore the relationship between the objects and the machine.

        You have **{horizon} steps** to complete the task. You can also exit the task early if you think you understand the relationship between the objects and the machine. After the task is done, you will be asked which objects are blickets, and the rule for turning on the machine.

        You will be prompted at each turn to choose actions.
        """)
    

    
    # Display environment description
    st.markdown("### Environment Description")
    st.markdown(game_state['feedback'])
    
    # Text-only version: Display action history
    if USE_TEXT_VERSION:
        st.markdown("### Action History")
        if st.session_state.action_history:
            for action_text in st.session_state.action_history:
                st.markdown(f"• {action_text}")
        else:
            st.markdown("*No actions taken yet.*")
        st.markdown("---")
    
    # Display the blicket machine (only in visual version)
    if not USE_TEXT_VERSION:
        st.markdown("### The Blicket Machine")
    
    # Determine if machine should be lit
    machine_lit = game_state['true_state'][-1]
    
    if not USE_TEXT_VERSION:
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
    
    # Text-only version: Show steps counter
    if USE_TEXT_VERSION:
        horizon = round_config.get('horizon', 32)
        steps_left = horizon - st.session_state.steps_taken
        st.markdown(f"**Steps Remaining: {steps_left}**")
        st.markdown("---")
    
    # Show warning if no steps left
    if steps_left <= 0:
        st.markdown("""
        <div style="background: rgba(220, 53, 69, 0.8); border: 1px solid rgba(220, 53, 69, 0.9); border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center;">
            <strong>⚠️ No steps remaining! You can only proceed to answer questions.</strong>
        </div>
        """, unsafe_allow_html=True)
    

    
    # Display available objects
    st.markdown("### Available Objects")
    
    if USE_TEXT_VERSION:
        st.markdown("Click on an object to place it on the machine. Click again to remove it.")
        
        # Text-only version: Simple button grid
        cols = st.columns(4)
        for i in range(round_config['num_objects']):
            with cols[i % 4]:
                is_selected = i in st.session_state.selected_objects
                horizon = round_config.get('horizon', 32)
                steps_left = horizon - st.session_state.steps_taken
                interaction_disabled = (steps_left <= 0 or st.session_state.visual_game_state == "questionnaire")
                
                # Create selection indicator box above the button
                box_color = "#00ff00" if is_selected else "#333333"
                st.markdown(f"""
                <div style="margin: 5px;">
                    <div style="
                        width: 60px; 
                        height: 20px; 
                        background-color: {box_color}; 
                        border: 2px solid #666666; 
                        border-radius: 5px; 
                        margin-bottom: 5px;
                        transition: all 0.2s ease;
                    "></div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Object {i + 1}", 
                           key=f"obj_{i}", 
                           disabled=interaction_disabled,
                           help=f"Click to {'remove' if is_selected else 'place'} Object {i + 1}"):
                    # Record the action before making changes
                    action_time = datetime.datetime.now()
                    action_type = "remove" if is_selected else "place"
                    
                    # Update object selection
                    if is_selected:
                        st.session_state.selected_objects.remove(i)
                    else:
                        st.session_state.selected_objects.add(i)
                    
                    # Update environment state
                    env._state[i] = (i in st.session_state.selected_objects)
                    env._update_machine_state()
                    game_state = env.step("look")[0]  # Get updated state
                    st.session_state.game_state = game_state
                    
                    # Add to action history
                    action_text = f"You {'removed' if action_type == 'remove' else 'put'} Object {i + 1} {'from' if action_type == 'remove' else 'on'} the machine. The blicket detector is {'LIT' if game_state['true_state'][-1] else 'NOT LIT'}."
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
                        "object_index": i,
                        "object_id": f"object_{i + 1}",
                        "machine_state_before": bool(not machine_lit),  # Previous state
                        "machine_state_after": bool(game_state['true_state'][-1]),  # New state
                        "objects_on_machine": list(st.session_state.selected_objects),
                        "step_number": st.session_state.steps_taken + 1
                    }
                    st.session_state.user_actions.append(action_data)
                    
                    # Increment step counter
                    st.session_state.steps_taken += 1
                    st.experimental_rerun()
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
                        obj_idx = row_objects[j]
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
                        
                        if st.button(f"Select Object {obj_idx + 1}", key=f"obj_{obj_idx}", help=f"Click to {'remove' if is_selected else 'place'} Object {obj_idx + 1}"):
                            # Record the action before making changes
                            action_time = datetime.datetime.now()
                            action_type = "remove" if is_selected else "place"
                            
                            # Update object selection
                            if is_selected:
                                st.session_state.selected_objects.remove(obj_idx)
                            else:
                                st.session_state.selected_objects.add(obj_idx)
                            
                            # Update environment state
                            env._state[obj_idx] = (obj_idx in st.session_state.selected_objects)
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
                                "object_index": obj_idx,
                                "object_id": f"object_{obj_idx + 1}",
                                "machine_state_before": bool(not machine_lit),  # Previous state
                                "machine_state_after": bool(game_state['true_state'][-1]),  # New state
                                "objects_on_machine": list(st.session_state.selected_objects),
                                "step_number": st.session_state.steps_taken + 1
                            }
                            st.session_state.user_actions.append(action_data)
                            
                            # Increment step counter
                            st.session_state.steps_taken += 1
                            st.experimental_rerun()
                    
                    
    

    
    # Phase transition buttons
    st.markdown("---")
    
    if st.session_state.visual_game_state == "exploration":
        horizon = round_config.get('horizon', 32)
        steps_left = horizon - st.session_state.steps_taken
        
        # Allow proceeding to questions if steps are exhausted or user chooses to
        if steps_left <= 0:
            
            if st.button("Proceed to Answer Questions"):
                st.session_state.visual_game_state = "questionnaire"
                st.experimental_rerun()
        else:
            if st.button("Ready to Answer Questions"):
                st.session_state.visual_game_state = "questionnaire"
                st.experimental_rerun()
    
    elif st.session_state.visual_game_state == "questionnaire":
        st.markdown("""
        <div style="padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
            <h3 style="margin: 0; text-align: center;">🎯 Blicket Classification</h3>
            <p style="margin: 10px 0 0 0; text-align: center; opacity: 0.9;">For each object, indicate whether you think it is a blicket or not:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create questionnaire with object images
        for i in range(round_config['num_objects']):
            if USE_TEXT_VERSION:
                # Text-only version: Simple text-based questionnaire
                st.markdown(f"**Object {i + 1}**")
                st.radio(
                    f"Is Object {i + 1} a blicket?",
                    ["Yes", "No"],
                    key=f"blicket_q_{i}"
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
                    key=f"blicket_q_{i}"
                )
                
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Add rule question
        st.markdown("---")
        st.markdown("### Rule Inference")
        st.markdown("Based on your observations, what do you think is the rule for how the blicket detector works?")
        rule_hypothesis = st.text_area(
            "What do you think is the rule?",
            placeholder="Describe your hypothesis about how the blicket detector determines when to light up...",
            height=100,
            key="rule_hypothesis"
        )
        
        # Navigation buttons
        # Show Next Round button for all rounds except the last one
        if current_round + 1 < total_rounds:
            # Check if rule hypothesis is provided
            rule_hypothesis = st.session_state.get("rule_hypothesis", "").strip()
            if not rule_hypothesis:
                st.warning("⚠️ Please provide your hypothesis about the rule before proceeding to the next round.")
            
            if st.button("Next Round", disabled=not rule_hypothesis):
                # Collect blicket classifications
                blicket_classifications = {}
                for i in range(round_config['num_objects']):
                    blicket_classifications[f"object_{i+1}"] = st.session_state.get(f"blicket_q_{i}", "No")
                
                # Get rule hypothesis
                rule_hypothesis = st.session_state.get("rule_hypothesis", "")
                
                # Save current round data with detailed action tracking
                round_data = {
                    "start_time": st.session_state.game_start_time.isoformat(),
                    "end_time": datetime.datetime.now().isoformat(),
                    "round_number": current_round + 1,
                    "round_config": round_config,
                    "user_actions": st.session_state.user_actions,  # All place/remove actions
                    "blicket_classifications": blicket_classifications,  # User's blicket answers
                    "rule_hypothesis": rule_hypothesis,  # User's rule hypothesis
                    "true_blicket_indices": convert_numpy_types(game_state['blicket_indices']),
                    "final_machine_state": bool(game_state['true_state'][-1]),
                    "total_steps_taken": st.session_state.steps_taken,
                    "final_objects_on_machine": list(st.session_state.selected_objects),
                    "rule": round_config['rule']
                }
                
                # Use the provided save function or default Firebase function
                if save_data_func:
                    save_data_func(participant_id, round_data)
                else:
                    save_game_data(participant_id, round_data)
                
                # Clear session state for next round
                st.session_state.pop("visual_game_state", None)
                st.session_state.pop("env", None)
                st.session_state.pop("game_state", None)
                st.session_state.pop("object_positions", None)
                st.session_state.pop("selected_objects", None)
                st.session_state.pop("blicket_answers", None)
                st.session_state.pop("game_start_time", None)
                st.session_state.pop("shape_images", None)
                st.session_state.pop("steps_taken", None)
                st.session_state.pop("user_actions", None)
                
                # Return to main app for next round
                st.session_state.phase = "next_round"
                st.experimental_rerun()
        else:
            # Show Finish Task button only on the last round
            # Check if rule hypothesis is provided
            rule_hypothesis = st.session_state.get("rule_hypothesis", "").strip()
            if not rule_hypothesis:
                st.warning("⚠️ Please provide your hypothesis about the rule before finishing the task.")
            
            if st.button("Finish Task", disabled=not rule_hypothesis):
                # Collect blicket classifications
                blicket_classifications = {}
                for i in range(round_config['num_objects']):
                    blicket_classifications[f"object_{i+1}"] = st.session_state.get(f"blicket_q_{i}", "No")
                
                # Get rule hypothesis
                rule_hypothesis = st.session_state.get("rule_hypothesis", "")
                
                # Save final round data with detailed action tracking
                round_data = {
                    "start_time": st.session_state.game_start_time.isoformat(),
                    "end_time": datetime.datetime.now().isoformat(),
                    "round_number": current_round + 1,
                    "round_config": round_config,
                    "user_actions": st.session_state.user_actions,  # All place/remove actions
                    "blicket_classifications": blicket_classifications,  # User's blicket answers
                    "rule_hypothesis": rule_hypothesis,  # User's rule hypothesis
                    "true_blicket_indices": convert_numpy_types(game_state['blicket_indices']),
                    "final_machine_state": bool(game_state['true_state'][-1]),
                    "total_steps_taken": st.session_state.steps_taken,
                    "final_objects_on_machine": list(st.session_state.selected_objects),
                    "rule": round_config['rule']
                }
                
                # Use the provided save function or default Firebase function
                if save_data_func:
                    save_data_func(participant_id, round_data)
                else:
                    save_game_data(participant_id, round_data)
                
                # Clear session state
                st.session_state.pop("visual_game_state", None)
                st.session_state.pop("env", None)
                st.session_state.pop("game_state", None)
                st.session_state.pop("object_positions", None)
                st.session_state.pop("selected_objects", None)
                st.session_state.pop("blicket_answers", None)
                st.session_state.pop("game_start_time", None)
                st.session_state.pop("shape_images", None)
                st.session_state.pop("steps_taken", None)
                st.session_state.pop("user_actions", None)
                
                # Return to main app for completion
                st.session_state.phase = "end"
                st.experimental_rerun()

if __name__ == "__main__":
    # Test the visual game
    test_config = {
        'num_objects': 4,
        'num_blickets': 2,
        'rule': 'conjunctive',
        'init_prob': 0.1,
        'transition_noise': 0.0
    }
    visual_blicket_game_page("test_user", test_config, 0, 3, None)
