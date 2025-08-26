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
    .blicket-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .machine-display {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .object-grid {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
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
        
        # Initialize fixed shape images for this round
        st.session_state.shape_images = []
        for i in range(round_config['num_objects']):
            shape_num = random.randint(1, 8)
            shape_path = f"images/shape{shape_num}.png"
            st.session_state.shape_images.append(get_image_base64(shape_path))
    
    # Get environment state
    env = st.session_state.env
    game_state = st.session_state.game_state
    
    # Load images
    blicket_img = get_image_base64("images/blicket.png")
    blicket_lit_img = get_image_base64("images/blicket_lit.png")
    
    # Use fixed shape images from session state
    shape_images = st.session_state.shape_images
    
    # Display round info and progress
    st.markdown(f"## Round {current_round + 1} of {total_rounds}")
    
    # Progress bar
    progress = (current_round + 1) / total_rounds
    st.progress(progress)
    
    # Display environment description
    st.markdown("### Environment Description")
    st.markdown(game_state['feedback'])
    
    # Display the blicket machine
    st.markdown("### The Blicket Machine")
    
    # Determine if machine should be lit
    machine_lit = game_state['true_state'][-1]
    machine_img = blicket_lit_img if machine_lit else blicket_img
    
    # Apply CSS class to machine section
    st.markdown('<div class="machine-display">', unsafe_allow_html=True)
    
    # Create machine display with objects on top
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <img src="data:image/png;base64,{machine_img}" style="width: 200px; height: auto;">
    </div>
    """, unsafe_allow_html=True)
    
    # Display objects on machine
    if st.session_state.selected_objects:
        st.markdown("### Objects on Machine:")
        
        # Create a horizontal layout for objects on machine
        objects_container = st.container()
        with objects_container:
            # Use HTML to create a horizontal layout
            objects_html = ""
            for obj_idx in st.session_state.selected_objects:
                objects_html += f"""
                <div style="display: inline-block; margin: 10px; text-align: center; vertical-align: top;">
                    <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 60px; height: auto;">
                    <br><strong>Object {obj_idx + 1}</strong>
                </div>
                """
            
            st.markdown(f"""
            <div style="text-align: center; margin: 10px 0;">
                {objects_html}
            </div>
            """, unsafe_allow_html=True)
    
    # Display available objects
    st.markdown("### Available Objects")
    st.markdown("Click on an object to place it on the machine. Click again to remove it.")
    
    # Apply CSS class to object grid section
    st.markdown('<div class="object-grid">', unsafe_allow_html=True)
    
    # Create grid of objects using Streamlit components
    for i in range(0, round_config['num_objects'], 4):
        # Create a row of up to 4 objects
        row_objects = range(i, min(i + 4, round_config['num_objects']))
        
        # Create columns for this row
        cols = st.columns(len(row_objects))
        
        for j, obj_idx in enumerate(row_objects):
            with cols[j]:
                # Check if object is currently selected
                is_selected = obj_idx in st.session_state.selected_objects
                
                # Display object image with selection indicator
                border_color = "2px solid #00ff00" if is_selected else "2px solid #cccccc"
                st.markdown(f"""
                <div style="text-align: center; margin: 10px; padding: 10px; border: {border_color}; border-radius: 10px;">
                    <img src="data:image/png;base64,{shape_images[obj_idx]}" style="width: 80px; height: auto;">
                    <br><strong>Object {obj_idx + 1}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Create clickable object button
                if st.button(f"Select Object {obj_idx + 1}", key=f"obj_{obj_idx}"):
                    if is_selected:
                        st.session_state.selected_objects.remove(obj_idx)
                    else:
                        st.session_state.selected_objects.add(obj_idx)
                    
                    # Update environment state
                    env._state[obj_idx] = (obj_idx in st.session_state.selected_objects)
                    env._update_machine_state()
                    game_state = env.step("look")[0]  # Get updated state
                    st.session_state.game_state = game_state
                    st.experimental_rerun()
    
    # Close object grid div
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Phase transition buttons
    st.markdown("---")
    
    if st.session_state.visual_game_state == "exploration":
        if st.button("Ready to Answer Questions"):
            st.session_state.visual_game_state = "questionnaire"
            st.experimental_rerun()
    
    elif st.session_state.visual_game_state == "questionnaire":
        st.markdown("### Blicket Classification")
        st.markdown("For each object, indicate whether you think it is a blicket or not:")
        
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
        if st.button("Back to Exploration"):
            st.session_state.visual_game_state = "exploration"
            st.experimental_rerun()
        
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
                        "true_blicket_indices": game_state['blicket_indices'].tolist(),
                        "final_machine_state": game_state['true_state'][-1]
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
                    
                    # Return to main app for next round
                    st.session_state.phase = "next_round"
                    st.experimental_rerun()
            else:
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
                        "true_blicket_indices": game_state['blicket_indices'].tolist(),
                        "final_machine_state": game_state['true_state'][-1]
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
