# Extension experiments

Four **separate** Streamlit Cloud deployments so the OG active and passive samples can
be expanded to ~400 each (conjunctive + disjunctive), all running concurrently. Each
link is pinned to one condition and writes to **its own Firebase** (set in that app's
own Streamlit Cloud secrets).

| Suggested Streamlit app name          | Main file path                                  | Condition             |
|---------------------------------------|-------------------------------------------------|-----------------------|
| `nexiom-text-game-active-conjunctive`  | `extension_experiments/active_conjunctive.py`   | Active, conjunctive   |
| `nexiom-text-game-active-disjunctive`  | `extension_experiments/active_disjunctive.py`   | Active, disjunctive   |
| `nexiom-text-game-passive-conjunctive` | `extension_experiments/passive_conjunctive.py`  | Passive, conjunctive  |
| `nexiom-text-game-passive-disjunctive` | `extension_experiments/passive_disjunctive.py`  | Passive, disjunctive  |

These wrappers reuse the existing `active_app/` and `passive_app/` code (no duplication);
they only pin the condition via env vars. The OG deployments are unchanged.

- **Active** links pin the main-game rule (`NEXIOM_MAIN_RULE`) instead of randomizing it.
- **Passive** links rotate respondents round-robin across the recorded action histories
  in a per-rule folder (see below).
- Every response is tagged with `condition` in Firebase, so each DB is self-describing.

There is **no participant cap in the app** — the "~400 total / +150 each" is a Prolific
recruitment number, managed on Prolific, not enforced here.

## Deploy each link on Streamlit Cloud

For each of the four apps:

1. **New app** from this repo/branch.
2. **Main file path** → the value from the table above.
3. **Advanced → Python requirements**: `extension_experiments/requirements.txt`.
4. **Secrets** → paste that app's own Firebase service account under `[firebase]`
   (same TOML shape as the root `STREAMLIT_CLOUD_SECRETS.md`), so this link writes to
   the new Firebase project you created for it. `database_url` must be the Realtime
   Database URL (`https://PROJECT-default-rtdb.firebaseio.com`).
5. **Passive links only** — tell it which histories to rotate over, one of:
   - drop the rule's `*.txt` action histories into
     `extension_experiments/histories/conjunctive/` or `.../disjunctive/`, **or**
   - set env var `NEXIOM_PASSIVE_HISTORY_DIR` to a folder, **or**
   - add a secret:
     ```toml
     [passive]
     history_dir = "extension_experiments/histories/disjunctive"
     ```
   Until a non-empty folder is found, the passive apps fall back to the single OG
   history so a link never breaks.
6. Deploy. Put the resulting URL into a Prolific study for that condition.

## Test locally first

```bash
./extension_experiments/run_all_local.sh
# active-conjunctive  :8511   active-disjunctive  :8512
# passive-conjunctive :8513   passive-disjunctive :8514
```

Locally all four use the repo's `.streamlit/secrets.toml`; on Cloud each uses its own.

## Action-history folders (passive)

`histories/conjunctive/` and `histories/disjunctive/` are where the recorded
action-history `.txt` files go — same format as
`active_explore/analysis/action_histories/*_action_history.txt`. Rotation is
round-robin across every `.txt` in the configured folder, coordinated by a Firebase
counter so concurrent respondents are distributed evenly.
