# Blicket Game - Version Guide

This project now features a **two-phase experimental design** with enhanced data collection.

## 🧠 Current Version

### Main App (`app.py`)
- **Best for research and data collection**
- **Two-phase design**: Comprehension + Main Experiment
- **Enhanced questionnaire** with rule type classification
- **Text-only interface** with 4 objects
- **Public URL**: https://blicket-text-game.streamlit.app/

**Features:**
- ✅ **Comprehension Phase**: Practice round (no data recorded)
- ✅ **Main Experiment**: 3 rounds with data collection
- ✅ **Enhanced Questions**: Rule inference + conjunctive/disjunctive classification
- ✅ **Clear Navigation**: Prominent buttons with helpful messages
- ✅ **Firebase Integration**: Automatic data saving
- ✅ **Fixed Configuration**: 4 objects for consistency

**Run locally:**
```bash
./run_app.sh
# or
streamlit run app.py
```

---

## 📊 Study Flow

### Phase 1: Participant ID Entry
- User enters participant ID
- System shows study overview with two phases

### Phase 2: Comprehension Phase
- **Practice round** to understand interface
- **No data recorded** - pure learning
- Simple conjunctive rule with 2 blickets
- Shorter time limit (16 steps)

### Phase 3: Main Experiment
- **3 rounds** with different configurations
- **All data recorded** to Firebase
- Random rules (conjunctive/disjunctive)
- Random number of blickets (1-4)
- Full time limit (32 steps)

### Phase 4: Enhanced Questionnaire
For each round, participants answer:
1. **Blicket Classification**: Which objects are blickets?
2. **Rule Inference**: Open-ended description of the rule
3. **Rule Type**: Multiple choice between conjunctive/disjunctive with explanations

---

## 🎯 Enhanced Data Collection

The new version collects richer data:

```json
{
  "interface_type": "text",
  "comprehension_completed": true,
  "rounds": [
    {
      "round_number": 1,
      "blicket_classifications": {...},
      "rule_hypothesis": "All blickets must be present...",
      "rule_type": "Conjunctive (ALL blickets must be present)",
      "user_actions": [...],
      "true_blicket_indices": [...],
      "rule": "conjunctive"
    }
  ]
}
```

**New Data Fields:**
- `rule_type`: Participant's conjunctive/disjunctive classification
- `comprehension_completed`: Whether practice phase was completed
- Enhanced rule hypothesis collection

---

## 🚀 Deployment

### For Public Sharing (Recommended)
Use the **main app** (`app.py`):
- **URL**: https://blicket-text-game.streamlit.app/
- Two-phase design ensures participants understand before data collection
- Enhanced questionnaire provides richer research data

### Firebase Setup
Follow the `STREAMLIT_CLOUD_SECRETS.md` guide to enable data collection in Streamlit Cloud.

---

## 🛠️ Technical Details

### File Structure
```
├── app.py                    # Main app with two-phase design
├── visual_blicket_game.py    # Game logic (supports practice mode)
├── run_app.sh               # Run main app
├── STREAMLIT_CLOUD_SECRETS.md # Firebase setup guide
└── ...
```

### Removed Files (Cleanup)
- `app_visual.py` - Redundant visual-only version
- `app_text.py` - Redundant text-only version  
- `run_visual.sh` - Redundant script
- `run_text.sh` - Redundant script

### Key Features
- **Practice Mode**: `is_practice=True` parameter prevents data saving
- **Enhanced Validation**: Both rule hypothesis AND rule type required
- **Clear Messaging**: Prominent warnings and instructions
- **Consistent Experience**: Fixed 4-object, text-only interface

---

## 🎯 Research Benefits

| Feature | Benefit |
|---------|---------|
| **Comprehension Phase** | Ensures participants understand before data collection |
| **Rule Type Question** | Quantitative measure of rule understanding |
| **Enhanced Validation** | Prevents incomplete responses |
| **Consistent Interface** | Eliminates interface effects on results |
| **Clear Navigation** | Reduces user confusion and dropouts |

---

## 🔧 Development Notes

The app now uses a single, streamlined codebase:
- **No interface selection** - fixed to text-only for consistency
- **No object count selection** - fixed to 4 objects for experimental control
- **Practice mode support** - `is_practice` parameter controls data saving
- **Enhanced questionnaire** - collects both qualitative and quantitative rule understanding

All navigation buttons are optimized for visibility and user experience.