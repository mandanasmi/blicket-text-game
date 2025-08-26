import os
import json
import random
import datetime
import streamlit as st
import numpy as np
from PIL import Image
import io
import base64

import env.blicket_text as blicket_text

def get_image_base64(image_path):
    """Convert image to base64 string for display"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

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
    # This would integrate with your existing Firebase setup
    # For now, we'll just log it
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
        
        # Initialize fixed shape images for this round (ensure different images)
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
    
    # Use fixed shape images from session state
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
    
    # Display the blicket machine
    st.markdown("### The Blicket Machine")
    
    # Determine if machine should be lit
    machine_lit = game_state['true_state'][-1]
    machine_img = blicket_lit_img if machine_lit else blicket_img
    

    
    # Create machine display with steps counter
    horizon = round_config.get('horizon', 32)
    steps_left = horizon - st.session_state.steps_taken
    
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
            <div>
                <img src="data:image/png;base64,{machine_img}" style="width: 200px; height: auto;">
            </div>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px 25px; border-radius: 15px; color: white; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                <div style="font-size: 14px; margin-bottom: 5px;">Steps Remaining</div>
                <div style="font-size: 24px;">{steps_left}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show warning if no steps left
    if steps_left <= 0:
        st.markdown("""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center;">
            <strong>⚠️ No steps remaining! You can only proceed to answer questions.</strong>
        </div>
        """, unsafe_allow_html=True)
    

    
    # Display available objects
    st.markdown("### Available Objects")
    st.markdown("Click on an object to place it on the machine. Click again to remove it.")
    

    
    # Create grid of objects with clickable images
    for i in range(0, round_config['num_objects'], 4):
        # Create a row of up to 4 objects
        row_objects = range(i, min(i + 4, round_config['num_objects']))
        
        # Create columns for this row
        cols = st.columns(len(row_objects))
        
        for j, obj_idx in enumerate(row_objects):
            with cols[j]:
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
                    border_color = "#00ff00" if is_selected else "#4CAF50"
                
                # Create clickable image container
                st.markdown(f"""
                <div style="text-align: center; margin: 10px;">
                    <div style="
                        display: inline-block; 
                        padding: 15px; 
                        border: 3px solid {border_color}; 
                        border-radius: 15px; 
                        background: {'rgba(0,255,0,0.1)' if is_selected else 'white'};
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                        transition: all 0.3s ease;
                        cursor: {cursor};
                        opacity: {opacity};
                    " onclick="document.getElementById('click_{obj_idx}').click()">
                        <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 80px; height: auto; margin-bottom: 10px;">
                        <br>
                        <div style="font-weight: bold; color: {'#00ff00' if is_selected else '#333'}; font-size: 16px;">
                            Object {obj_idx + 1}
                        </div>
                        <div style="font-size: 12px; color: #666; margin-top: 5px;">
                            {'Selected' if is_selected else 'Click to select'}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Hidden button for handling clicks
                if st.button("", key=f"click_{obj_idx}", help=f"Click Object {obj_idx + 1}"):
                    if not interaction_disabled:
                        if is_selected:
                            st.session_state.selected_objects.remove(obj_idx)
                        else:
                            st.session_state.selected_objects.add(obj_idx)
                        
                        # Update environment state
                        env._state[obj_idx] = (obj_idx in st.session_state.selected_objects)
                        env._update_machine_state()
                        game_state = env.step("look")[0]  # Get updated state
                        st.session_state.game_state = game_state
                        
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
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 15px; margin: 10px 0;">
                    <strong>No steps remaining. You must proceed to answer questions.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Proceed to Answer Questions"):
                st.session_state.visual_game_state = "questionnaire"
                st.experimental_rerun()
        else:
            st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <div style="background: #e8f5e8; border: 1px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 10px 0;">
                    <strong>Ready to test your understanding?</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Ready to Answer Questions"):
                st.session_state.visual_game_state = "questionnaire"
                st.experimental_rerun()
    
    elif st.session_state.visual_game_state == "questionnaire":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
            <h3 style="margin: 0; text-align: center;">🎯 Blicket Classification</h3>
            <p style="margin: 10px 0 0 0; text-align: center; opacity: 0.9;">For each object, indicate whether you think it is a blicket or not:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create questionnaire with object images
        for i in range(round_config['num_objects']):
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                <div style="flex: 0 0 100px; text-align: center;">
                    <img src="data:image/png;base64,{shape_images[i]}" style="width: 60px; height: auto;">
                    <br><strong>Object {i + 1}</strong>
                </div>
                <div style="flex: 1; margin-left: 20px;">
            """, unsafe_allow_html=True)
            
            st.radio(
                f"Is Object {i + 1} a blicket?",
                ["Yes", "No"],
                key=f"blicket_q_{i}"
            )
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Navigation buttons
        # Show Next Round button for all rounds except the last one
        if current_round + 1 < total_rounds:
            if st.button("Next Round"):
                # Save current round data
                round_data = {
                    "start_time": st.session_state.game_start_time.isoformat(),
                    "round_number": current_round + 1,
                    "round_config": round_config,
                    "blicket_answers": {
                        f"object_{i+1}": st.session_state.get(f"blicket_q_{i}", "No")
                        for i in range(round_config['num_objects'])
                    },
                    "true_blicket_indices": [int(x) for x in game_state['blicket_indices']] if isinstance(game_state['blicket_indices'], list) else [int(x) for x in game_state['blicket_indices'].tolist()],
                    "final_machine_state": bool(game_state['true_state'][-1])
                }
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
                
                # Return to main app for next round
                st.session_state.phase = "next_round"
                st.experimental_rerun()
        else:
            # Show Finish Task button only on the last round
            if st.button("Finish Task"):
                # Save final round data
                round_data = {
                    "start_time": st.session_state.game_start_time.isoformat(),
                    "round_number": current_round + 1,
                    "round_config": round_config,
                    "blicket_answers": {
                        f"object_{i+1}": st.session_state.get(f"blicket_q_{i}", "No")
                        for i in range(round_config['num_objects'])
                    },
                    "true_blicket_indices": [int(x) for x in game_state['blicket_indices']] if isinstance(game_state['blicket_indices'], list) else [int(x) for x in game_state['blicket_indices'].tolist()],
                    "final_machine_state": bool(game_state['true_state'][-1])
                }
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
