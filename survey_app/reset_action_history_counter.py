"""
Reset the action-history assignment counter in Firebase to 0.
Run this once before starting Prolific data collection so participants
get action histories in order from file 0, 1, 2, ... 101.

Usage (from project root or survey_app):
  Set Firebase env vars (FIREBASE_PROJECT_ID, FIREBASE_DATABASE_URL, etc.)
  then:
    python survey_app/reset_action_history_counter.py

Or with .env in survey_app or project root:
    python survey_app/reset_action_history_counter.py
"""
import os
import sys

# Load .env from survey_app or project root
_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_dir, "..")
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_dir, ".env"))
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# Optional: load from .streamlit/secrets.toml (same as Streamlit app when run locally)
def _load_secrets_toml():
    toml_path = os.path.join(_root, ".streamlit", "secrets.toml")
    if not os.path.isfile(toml_path):
        return
    try:
        import tomllib
    except ImportError:
        try:
            import toml as tomllib
        except ImportError:
            return
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    fb = data.get("firebase", {})
    if not fb:
        return
    key_to_env = {
        "project_id": "FIREBASE_PROJECT_ID",
        "private_key_id": "FIREBASE_PRIVATE_KEY_ID",
        "private_key": "FIREBASE_PRIVATE_KEY",
        "client_email": "FIREBASE_CLIENT_EMAIL",
        "client_id": "FIREBASE_CLIENT_ID",
        "client_x509_cert_url": "FIREBASE_CLIENT_X509_CERT_URL",
        "database_url": "FIREBASE_DATABASE_URL",
    }
    for k, env_key in key_to_env.items():
        v = fb.get(k)
        if v and not os.getenv(env_key):
            os.environ[env_key] = str(v).strip() if isinstance(v, str) else str(v)

_load_secrets_toml()

import firebase_admin
from firebase_admin import credentials, db

def main():
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    database_url = os.getenv("FIREBASE_DATABASE_URL")
    if not project_id or not database_url:
        print("Set FIREBASE_PROJECT_ID and FIREBASE_DATABASE_URL (env or .streamlit/secrets.toml). See survey_app/FIREBASE.md and SECRETS_TOML.md.")
        sys.exit(1)

    cred_dict = {
        "type": "service_account",
        "project_id": project_id,
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": (os.getenv("FIREBASE_PRIVATE_KEY") or "").replace("\\n", "\n"),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
        "universe_domain": "googleapis.com",
    }
    if not cred_dict.get("private_key") or not cred_dict.get("client_email"):
        print("Set FIREBASE_PRIVATE_KEY and FIREBASE_CLIENT_EMAIL (env or .streamlit/secrets.toml).")
        sys.exit(1)

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})

    ref = db.reference()
    config_ref = ref.child("_config").child("action_history_next_index")
    config_ref.set(0)
    print("Reset _config/action_history_next_index to 0. Next survey use will get action history index 0.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
