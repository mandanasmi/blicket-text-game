"""
Reset the varied-Nexiom assignment counter in Firebase to 0.
Run this before starting (or restarting) data collection on act-noc / act-sid, or after
changing NEXIOM_VARIED_OBJECT_COUNTS, so the round-robin starts from the first setup.

Each extension app has its own Firebase, so point the script at that app's secrets:
    python active_app/reset_varied_nexiom_counter.py .streamlit/secrets.act_noc.toml
    python active_app/reset_varied_nexiom_counter.py .streamlit/secrets.act_sid.toml

Several secrets files can be given at once. The file must have a [firebase] block
(same TOML shape as the Streamlit Cloud secrets).
"""
import sys

import firebase_admin
from firebase_admin import credentials, db

try:
    import tomllib

    def _load_toml(path):
        with open(path, "rb") as f:
            return tomllib.load(f)
except ImportError:
    import toml

    def _load_toml(path):
        return toml.load(path)


COUNTER_PATH = ("_config", "varied_nexiom_next_index")


def reset_counter(secrets_path):
    fb = _load_toml(secrets_path).get("firebase", {})
    database_url = fb.get("database_url")
    if not fb.get("project_id") or not database_url or not fb.get("private_key"):
        print(f"{secrets_path}: missing project_id, database_url or private_key in [firebase].")
        return False

    cred_dict = {
        "type": "service_account",
        "project_id": fb["project_id"],
        "private_key_id": fb.get("private_key_id"),
        "private_key": fb["private_key"].replace("\\n", "\n"),
        "client_email": fb.get("client_email"),
        "client_id": fb.get("client_id"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": fb.get("client_x509_cert_url"),
        "universe_domain": "googleapis.com",
    }
    # One named Firebase app per secrets file, so several databases can be reset in one run.
    app = firebase_admin.initialize_app(
        credentials.Certificate(cred_dict), {"databaseURL": database_url}, name=secrets_path
    )
    ref = db.reference("/".join(COUNTER_PATH), app=app)
    previous = ref.get()
    ref.set(0)
    print(f"{fb['project_id']}: reset {'/'.join(COUNTER_PATH)} from {previous} to 0.")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ok = all([reset_counter(path) for path in sys.argv[1:]])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
