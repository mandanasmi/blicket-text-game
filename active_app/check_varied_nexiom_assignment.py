"""
Check that the round-robin Nexiom assignment on act-noc / act-sid is staying balanced.

For each app's Firebase it reports the counter value, how many participants were
assigned to / completed each (num_objects, num_nexioms) setup, how often each object
was a Nexiom, and any gaps in the assignment indices (people who got an index but never
had their config saved).

    python active_app/check_varied_nexiom_assignment.py .streamlit/secrets.act_noc.toml .streamlit/secrets.act_sid.toml

"Assigned" = config saved when they clicked Start Main Experiment; "completed" = their
main_game record exists (written only when they finish the game and questions).
Participants without a varied_assignment_index (from before the rotation) are skipped.
"""
import sys
from collections import Counter

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


def _connect(secrets_path):
    fb = _load_toml(secrets_path)["firebase"]
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
    app = firebase_admin.initialize_app(
        credentials.Certificate(cred_dict), {"databaseURL": fb["database_url"]}, name=secrets_path
    )
    return fb["project_id"], app


def check(secrets_path):
    project_id, app = _connect(secrets_path)
    data = db.reference("/", app=app).get() or {}
    counter = (data.get("_config") or {}).get("varied_nexiom_next_index")

    assigned, completed = Counter(), Counter()
    nexiom_counts = {}  # setup -> Counter of 1-based object labels (completed only)
    indices, rules, skipped = [], Counter(), 0
    for key, node in data.items():
        if key.startswith("_") or not isinstance(node, dict):
            continue
        config = node.get("config") or {}
        index = config.get("varied_assignment_index")
        rounds = config.get("rounds") or []
        if index is None or not rounds:
            skipped += 1
            continue
        round_config = rounds[0]
        nexioms = round_config.get("blicket_indices") or []
        setup = (round_config["num_objects"], len(nexioms))
        indices.append(index)
        rules[round_config.get("rule")] += 1
        assigned[setup] += 1
        if node.get("main_game"):
            completed[setup] += 1
            nexiom_counts.setdefault(setup, Counter()).update(i + 1 for i in nexioms)

    print(f"\n=== {project_id} ===")
    print(f"Counter (_config/varied_nexiom_next_index): {counter}")
    print(f"Participants with an assignment: {len(indices)}  (rule: {dict(rules)})")
    if skipped:
        print(f"Skipped {skipped} participant(s) without a varied assignment (pre-rotation or not started).")
    if not indices:
        return

    print(f"\n{'Setup':<10}{'Assigned':>10}{'Completed':>11}{'Completion':>12}")
    for setup in sorted(assigned):
        a, c = assigned[setup], completed[setup]
        print(f"{f'{setup[1]}/{setup[0]}':<10}{a:>10}{c:>11}{c / a:>12.0%}")
    done = [completed[s] for s in assigned]
    print(f"Completed spread (max - min): {max(done) - min(done)}")

    for setup in sorted(nexiom_counts):
        if setup[1] == setup[0]:
            continue  # every object is a Nexiom, nothing to balance
        counts = nexiom_counts[setup]
        row = "  ".join(f"{obj}:{counts.get(obj, 0)}" for obj in range(1, setup[0] + 1))
        print(f"Nexiom frequency by object, {setup[1]}/{setup[0]} completed: {row}")

    duplicates = [i for i, n in Counter(indices).items() if n > 1]
    if duplicates:
        print(f"WARNING duplicate assignment indices: {sorted(duplicates)}")
    if counter is not None:
        # Indices handed out by the counter but never saved to a participant config.
        missing = sorted(set(range(counter)) - set(indices))
        if missing:
            print(f"Indices handed out but never saved ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for path in sys.argv[1:]:
        check(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
