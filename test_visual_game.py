#!/usr/bin/env python3
"""
Test script for the visual blicket game
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_blicket_game import create_new_game, get_image_base64

def test_basic_functionality():
    """Test basic game creation and image loading"""
    print("Testing basic functionality...")
    
    # Test game creation
    try:
        env, game_state = create_new_game(
            num_objects=4,
            num_blickets=2,
            rule="conjunctive",
            seed=42
        )
        print("✓ Game creation successful")
        print(f"  - Number of objects: {env.num_objects}")
        print(f"  - Number of blickets: {env.num_blickets}")
        print(f"  - Rule: {env.rule}")
        print(f"  - Blicket indices: {env.blicket_indices}")
    except Exception as e:
        print(f"✗ Game creation failed: {e}")
        return False
    
    # Test image loading
    try:
        blicket_img = get_image_base64("images/blicket.png")
        blicket_lit_img = get_image_base64("images/blicket_lit.png")
        print("✓ Image loading successful")
        print(f"  - Blicket image size: {len(blicket_img)} chars")
        print(f"  - Blicket lit image size: {len(blicket_lit_img)} chars")
    except Exception as e:
        print(f"✗ Image loading failed: {e}")
        return False
    
    # Test shape image loading
    try:
        shape_images = []
        for i in range(4):
            shape_num = (i % 8) + 1
            shape_path = f"images/shape{shape_num}.png"
            shape_img = get_image_base64(shape_path)
            shape_images.append(shape_img)
        print("✓ Shape image loading successful")
        print(f"  - Loaded {len(shape_images)} shape images")
    except Exception as e:
        print(f"✗ Shape image loading failed: {e}")
        return False
    
    # Test environment interaction
    try:
        # Test putting an object on the machine
        env._state[0] = True  # Put first object on machine
        env._update_machine_state()
        new_state = env.step("look")[0]
        print("✓ Environment interaction successful")
        print(f"  - Machine state after putting object 0: {new_state['true_state'][-1]}")
    except Exception as e:
        print(f"✗ Environment interaction failed: {e}")
        return False
    
    print("\nAll tests passed! ✓")
    return True

if __name__ == "__main__":
    test_basic_functionality()
