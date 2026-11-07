# Extension experiments

## Active mixed-rule study

The existing active conjunctive link now uses `(A AND B) OR C` (`mixed_and_or`);
the active disjunctive link uses `(A OR B) AND C` (`mixed_or_and`). Both have
8 objects, 3 Nexioms and 32 tests. Nexiom identities rotate; A/B/C roles are
shuffled per participant and preserved in the ordered `blicket_indices` config
(first two are A/B, third is C; config uses zero-based indices).
Saved conditions are `active_mixed_and_or` and `active_mixed_or_and` so new
responses can be distinguished from previous pure-rule data in the same databases.
The rule question offers Conjunctive, Disjunctive and Mixed without examples;
Mixed is the correct rule-type answer for both links. Intervention and bonus
grading use the corresponding mixed rule. Passive links retain their pure rules.
Deploy the updated code to activate these changes; existing participant sessions
retain their saved round configuration.

## Deployment setup

Four **separate** Streamlit Cloud deployments so the OG active and passive samples can
be expanded to ~400 each (conjunctive + disjunctive), all running concurrently. Each
link is pinned to one condition and writes to **its own Firebase** (set in that app's
own Streamlit Cloud secrets).

| Suggested Streamlit app name          | Main file path                                  | Condition             |
|---------------------------------------|-------------------------------------------------|-----------------------|
| `nexiom-text-game-active-conjunctive`  | `extension_experiments/active_conjunctive.py`   | Active, mixed AND/OR |
| `nexiom-text-game-active-disjunctive`  | `extension_experiments/active_disjunctive.py`   | Active, mixed OR/AND |
| `nexiom-text-game-passive-conjunctive` | `extension_experiments/passive_conjunctive.py`  | Passive, conjunctive  |
| `nexiom-text-game-passive-disjunctive` | `extension_experiments/passive_disjunctive.py`  | Passive, disjunctive  |

These wrappers reuse the existing `active_app/` and `passive_app/` code (no duplication);
they pin the condition via env vars. The OG deployments are unchanged.

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

## AI feedback (act-sid only)

On the active disjunctive link (act-sid, `active_disjunctive.py`, mixed `(A OR B) AND C`)
participants can ask an AI (OpenAI, default `gpt-5.5`) for feedback during main-game
exploration, optionally typing a question. The act-noc link sets `NEXIOM_AI_FEEDBACK=0`
and is unchanged.

- The AI sees only the task description and the participant's test history. It is never
  given the true Nexioms or rule. Prompt and logic: `active_app/ai_feedback.py`.
- Each exchange (question, exact prompt sent, reply, model, latency, tests done when
  asked) is saved right away under `<pid>/ai_feedback/<n>` and copied into the main-round
  record as `ai_feedback_log` / `ai_feedback_requests`. `config.ai_feedback` is `true` for
  participants who had the panel.
- Add the API key to act-sid's Streamlit Cloud secrets:
  ```toml
  [openai]
  api_key = "sk-proj-..."
  ```
  (or set `OPENAI_API_KEY`). Without a key the panel shows an error and the game still works.
- Optional env/secrets: `NEXIOM_AI_FEEDBACK_MODEL` (default `gpt-5.5`),
  `NEXIOM_AI_FEEDBACK_EFFORT` (reasoning effort, default `low`) and
  `NEXIOM_AI_FEEDBACK_MAX` (answered requests per game, default 10; 0 = no limit).

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

## Proctoring (active links)

The two active wrappers set `NEXIOM_PROCTORED=1` (see `active_app/proctoring.py`). After
consent, participants see a "Before you begin" page (stay in full screen or risk losing
compensation; bonus for identifying all Nexioms **and** the correct rule type), then must
enter full screen. The test history stays visible during Q&A (`NEXIOM_HIDE_HISTORY_IN_QA=0`).

- A browser cannot block screenshots or a second device, so this is a deterrent plus
  logging: leaving full screen or switching tab/window shows a blocking overlay; copy,
  cut, paste, right-click and Ctrl/Cmd+P are blocked; PrintScreen presses are logged.
- Everything is saved to Firebase under `<pid>/integrity` (counts, `away_seconds`,
  per-phase breakdown, event log), and a summary is copied into the main-round record
  next to `bonus_eligible`, `bonus_nexioms_correct`, `bonus_rule_type_correct`.
- `config.proctored` / `config.history_shown_in_qa` mark which participants ran this way.
- If full screen is unavailable (e.g. the Cloud wrapper iframe blocks it),
  `fullscreen_supported` is `false` and only focus tracking runs.
