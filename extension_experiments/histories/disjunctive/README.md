# Disjunctive action histories (passive-disjunctive)

Drop the recorded **disjunctive** action-history `.txt` files here (same format as
`active_explore/analysis/action_histories/*_action_history.txt`). The
`passive-disjunctive` link rotates respondents round-robin across every `.txt`
file in this folder.

While this folder is empty, the passive-disjunctive app falls back to the single
OG conjunctive history so it never breaks. Populate before launch, or instead point
`NEXIOM_PASSIVE_HISTORY_DIR` / `[passive].history_dir` at your own folder.
