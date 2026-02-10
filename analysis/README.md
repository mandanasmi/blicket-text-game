# Analysis for 10 Prolific Participants

Simple analysis scripts for your Nexiom Text Game data.

## Quick Start (3 Steps)

### Step 1: Export Your Data

1. Go to: https://console.firebase.google.com/u/0/project/nexiom-text-game/database/nexiom-text-game-default-rtdb/data
2. Click **⋮** (three dots) → **Export JSON**
3. Save as: `firebase_data.json` in this folder

### Step 2: Run Analysis

```bash
cd /home/mila/s/samieima/projects/blicket-text-game/analysis

# Analyze data (creates CSV files)
python3 analyze_from_json.py

# Generate visualizations (creates PNG files)
python3 visualize_results.py
```

### Step 3: View Results

You'll get **4 PNG visualizations** and **3 CSV files** with your analysis.

---

## What Gets Analyzed

The scripts analyze your 10 Prolific participants (24-character hex IDs like `5c4b987538878c0001c7883b`) and answer three research questions:

### 1. Comprehension Phase Correctness
- Did participants correctly identify Nexioms in practice?
- Percentage who got it right

### 2. Exploration Behavior  
- Average and range of tests per session
- How many participants tested more than 4 times
- Breakdown by phase and round

### 3. Rule Understanding
- Accuracy of conjunctive vs disjunctive classification
- Blicket identification accuracy
- Performance differences between rule types

---

## Output Files

**PNG Visualizations:**
- `comprehension_correctness.png` - Bar charts of correctness
- `exploration_statistics.png` - Test behavior patterns and distributions
- `rule_understanding.png` - Classification accuracy by rule type
- `confusion_matrix.png` - Classification error patterns

**CSV Data Files:**
- `comprehension_analysis.csv` - Each participant's comprehension data
- `exploration_analysis.csv` - Test counts for each session
- `rule_understanding_analysis.csv` - Rule classification results

---

## Participant Filtering

The scripts automatically:
- **Include:** 24-character hexadecimal Prolific IDs
- **Exclude:** Test users (IDs containing: test, demo, alice, bob, etc.)

Expected: 10 Prolific participants

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

Or:
```bash
pip install -r requirements.txt
```

---

## Files in This Folder

- **analyze_from_json.py** - Main analysis script (works with exported JSON)
- **visualize_results.py** - Creates PNG charts from CSV results
- **requirements.txt** - Python dependencies
- **README.md** - This file

---

## Troubleshooting

**"File not found: firebase_data.json"**
- Make sure you exported your Firebase data as JSON
- Save it as `firebase_data.json` in this folder

**"No Prolific participants found"**
- Check that your participant IDs are 24-character hex strings
- The script will show which IDs were excluded

**"Module not found"**
- Install dependencies: `pip install pandas numpy matplotlib seaborn`

---

## Need Help?

If you encounter issues:
1. Check that `firebase_data.json` exists in this folder
2. Verify the file contains valid JSON data
3. Make sure Python packages are installed
4. Check the terminal output for specific error messages
