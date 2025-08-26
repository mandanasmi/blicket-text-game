# Visual Blicket Game

This document describes the visual blicket game implementation that integrates with the existing text-based blicket environment.

## Overview

The visual blicket game provides a graphical interface for the blicket detection task, replacing the text-based command interface with clickable objects and visual feedback.

## Features

### Visual Interface
- **Blicket Machine Display**: Shows the blicket machine at the top with visual feedback (lit/unlit)
- **Object Grid**: Displays available objects as clickable images with numbered labels
- **Interactive Placement**: Click objects to place them on the machine, click again to remove
- **Real-time Feedback**: Machine state updates immediately when objects are added/removed
- **Visual Selection**: Selected objects are highlighted with green borders

### Game Flow
1. **Exploration Phase**: Users can experiment by placing objects on the machine
2. **Questionnaire Phase**: Users classify each object as a blicket or not
3. **Round Navigation**: Automatic progression through multiple rounds
4. **Data Collection**: All interactions and responses are saved to Firebase

## File Structure

```
blicket-text-game/
├── app.py                    # Main Streamlit application
├── visual_blicket_game.py    # Visual game interface
├── env/blicket_text.py       # Core blicket environment
├── images/                   # Game assets
│   ├── blicket.png          # Unlit blicket machine
│   ├── blicket_lit.png      # Lit blicket machine
│   └── shape1.png - shape8.png  # Object shapes
└── test_visual_game.py      # Test script
```

## How It Works

### Integration with Existing Environment
The visual game uses the same `BlicketTextEnv` class from `env/blicket_text.py`, ensuring:
- Consistent game logic and rules
- Same blicket detection algorithms (conjunctive/disjunctive)
- Compatible data structures and state management

### Visual Representation
- **Machine**: Uses `blicket.png` when off, `blicket_lit.png` when on
- **Objects**: Randomly selects from `shape1.png` through `shape8.png` for each object
- **Layout**: Responsive grid layout that adapts to different numbers of objects

### State Management
- **Session State**: Tracks current game state, selected objects, and user responses
- **Environment State**: Maintains the underlying blicket environment state
- **Round Progression**: Handles transitions between exploration and questionnaire phases

## Usage

### Running the Application
```bash
streamlit run app.py
```

### Game Flow
1. **Enter Participant ID**: User enters their unique identifier
2. **Configuration**: System generates random game configurations for multiple rounds
3. **Visual Game**: For each round:
   - View environment description
   - Click objects to place them on the machine
   - Observe machine state changes
   - Answer blicket classification questions
4. **Data Collection**: All responses are automatically saved to Firebase

### Testing
```bash
python test_visual_game.py
```

## Technical Details

### Image Handling
- Images are converted to base64 strings for inline display in Streamlit
- Random selection ensures variety in object appearances
- Responsive sizing adapts to different screen sizes

### State Synchronization
- Visual interface directly manipulates the underlying environment state
- Machine state updates are calculated using the same rules as the text version
- All interactions are logged for analysis

### Data Collection
- **Exploration Data**: Object placements, machine state changes, timing
- **Classification Data**: User's blicket/non-blicket judgments for each object
- **Ground Truth**: Actual blicket indices and machine rules for validation

## Customization

### Adding New Objects
1. Add new shape images to the `images/` directory
2. Update the shape selection logic in `visual_blicket_game.py`
3. Ensure images are properly sized and formatted

### Modifying Game Rules
- Changes to `env/blicket_text.py` automatically apply to the visual interface
- New blicket functions can be added to `BlicketFunctionSet`
- Environment parameters can be adjusted in the configuration generation

### Styling
- CSS classes are defined in the `visual_blicket_game_page` function
- Layout can be customized by modifying the HTML/CSS sections
- Responsive design adapts to different screen sizes

## Dependencies

- **Streamlit**: Web interface framework
- **NumPy**: Numerical computations
- **PIL**: Image processing (if needed for future enhancements)
- **Firebase**: Data storage and retrieval

## Future Enhancements

- **Animation**: Smooth transitions when objects are placed/removed
- **Sound Effects**: Audio feedback for machine state changes
- **Advanced Visualizations**: Heat maps showing object interaction patterns
- **Mobile Optimization**: Touch-friendly interface for mobile devices
- **Accessibility**: Screen reader support and keyboard navigation
