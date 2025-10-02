# Blicket Game - Version Guide

This project now offers **three different ways** to run the Blicket experiment:

## 📱 Available Versions

### 1. 🧙 **Main App with Interface Selection** (`app.py`)
- **Best for most users**
- Allows participants to choose between Visual or Text interface
- Single deployment with both options
- **Public URL**: https://blicket-text-game.streamlit.app/

**Features:**
- Interface selection screen
- Switch between modes anytime
- Unified data collection
- Best user experience

**Run locally:**
```bash
./run_app.sh
# or
streamlit run app.py
```

---

### 2. 🎨 **Visual-Only Version** (`app_visual.py`)
- **For visual interface only**
- Object images and shape representations
- Visual blicket detector machine
- Interactive visual elements

**Features:**
- ✅ Object images with unique shapes
- ✅ Visual blicket detector (lights up)
- ✅ Clickable object interactions
- ✅ Visual state history
- ✅ Rich visual feedback

**Run locally:**
```bash
./run_visual.sh
# or
streamlit run app_visual.py
```

---

### 3. 📝 **Text-Only Version** (`app_text.py`)
- **For text interface only**
- No images or visual elements
- Pure text descriptions
- Faster loading, better accessibility

**Features:**
- ✅ Text-based object representations ("Object 1", "Object 2", etc.)
- ✅ Text machine status ("🟢 LIT" or "🔴 NOT LIT")
- ✅ Simple button interactions
- ✅ Text-based state history
- ✅ Accessibility-friendly

**Run locally:**
```bash
./run_text.sh
# or
streamlit run app_text.py
```

---

## 🚀 Deployment Options

### For Public Sharing (Recommended)
Use the **main app** (`app.py`) which includes both interfaces:
- **URL**: https://blicket-text-game.streamlit.app/
- Participants can choose their preferred interface
- Single URL to share with friends

### For Specific Research Needs
Deploy individual versions if you need to control the interface:
- **Visual only**: Deploy `app_visual.py` 
- **Text only**: Deploy `app_text.py`

---

## 📊 Data Collection

All versions save the same data structure to Firebase, with an additional field:
```json
{
  "interface_type": "visual" | "text",
  "participant_actions": [...],
  "blicket_classifications": {...},
  "rule_hypothesis": "...",
  ...
}
```

This allows you to analyze differences between interface types in your research.

---

## 🛠️ Technical Details

### File Structure
```
├── app.py              # Main app with interface selection
├── app_visual.py       # Visual-only version
├── app_text.py         # Text-only version
├── visual_blicket_game.py  # Shared game logic (supports both modes)
├── run_app.sh          # Run main app
├── run_visual.sh       # Run visual app
├── run_text.sh         # Run text app
└── ...
```

### Visual vs Text Mode Logic
The `visual_blicket_game.py` file now supports both modes through the `use_visual_mode` parameter:
- `use_visual_mode=True`: Shows images, visual machine, rich UI
- `use_visual_mode=False`: Shows text only, simple buttons, text status

---

## 🎯 Which Version Should You Use?

| Use Case | Recommended Version | Why |
|----------|-------------------|-----|
| **General Research** | Main App (`app.py`) | Participants can choose, more data |
| **Visual-focused Study** | Visual App (`app_visual.py`) | Consistent visual experience |
| **Accessibility Study** | Text App (`app_text.py`) | No visual dependencies |
| **Sharing with Friends** | Main App (`app.py`) | Most flexible and user-friendly |

---

## 🔧 Development

All versions share the same core logic but differ in:
- **Interface rendering**: Visual vs text elements
- **User interaction**: Image clicks vs button clicks  
- **Data labeling**: Interface type tracking
- **Performance**: Visual version loads images, text version is faster

The shared `visual_blicket_game.py` handles both modes seamlessly.
