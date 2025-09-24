# Blicket Text Game - Setup Instructions

## Virtual Environment Setup

A Python virtual environment has been created and configured for this project.

### Quick Start

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Configure Firebase (Required):**
   - Copy `.env.example` to `.env`
   - Fill in your Firebase project credentials in the `.env` file
   - You'll need a Firebase project with Realtime Database enabled

3. **Run the application:**
   ```bash
   ./run_app.sh
   ```
   Or manually:
   ```bash
   source venv/bin/activate
   streamlit run app.py
   ```

### What's Installed

The virtual environment includes all required dependencies:
- **Streamlit** (1.50.0) - Web app framework
- **Firebase Admin** (7.1.0) - Firebase integration
- **NumPy** (1.26.4) - Numerical computing
- **Pandas** (1.5.3) - Data manipulation
- **Python-dotenv** (1.1.1) - Environment variable management
- **Watchdog** (6.0.0) - File system monitoring

### Firebase Configuration

The application requires Firebase credentials to save game data. You need to:

1. Create a Firebase project at https://console.firebase.google.com/
2. Enable Realtime Database
3. Create a service account and download the credentials
4. Add the credentials to your `.env` file

### Troubleshooting

- **Import errors**: Make sure the virtual environment is activated
- **Firebase errors**: Check your `.env` file configuration
- **Port conflicts**: Streamlit runs on port 8501 by default

### Deactivating the Virtual Environment

When you're done working:
```bash
deactivate
```
