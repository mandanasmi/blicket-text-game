"""Mean time per step: training vs test (disjunctive passive UI)."""
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "new-data-history.json")

with open(data_path) as f:
    data = json.load(f)

train_rows = []
test_rows = []
for record in data.values():
    if record.get("status") != "completed":
        continue
    d = record.get("disjunctive_case_data", {})
    tr = d.get("time_per_training_step_seconds")
    te = d.get("time_per_test_step_seconds")
    if not tr or not te or len(tr) != 6 or len(te) != 6:
        continue
    train_rows.append([float(x) for x in tr])
    test_rows.append([float(x) for x in te])

train = np.array(train_rows)
test = np.array(test_rows)
n = train.shape[0]

mean_tr = train.mean(axis=0)
mean_te = test.mean(axis=0)
sem_tr = train.std(axis=0, ddof=1) / math.sqrt(n)
sem_te = test.std(axis=0, ddof=1) / math.sqrt(n)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(6)
w = 0.36
bars_tr = ax.bar(
    x - w / 2,
    mean_tr,
    width=w,
    label="Training",
    color="#5DCAA2",
    edgecolor="#0F6E56",
    linewidth=1,
    zorder=3,
)
bars_te = ax.bar(
    x + w / 2,
    mean_te,
    width=w,
    label="Test",
    color="#9B8FD9",
    edgecolor="#3C3489",
    linewidth=1,
    zorder=3,
)

ax.errorbar(
    x - w / 2,
    mean_tr,
    yerr=sem_tr,
    fmt="none",
    color="#0F6E56",
    capsize=4,
    capthick=1.5,
    elinewidth=1.5,
    zorder=4,
)
ax.errorbar(
    x + w / 2,
    mean_te,
    yerr=sem_te,
    fmt="none",
    color="#3C3489",
    capsize=4,
    capthick=1.5,
    elinewidth=1.5,
    zorder=4,
)

ax.set_xticks(x)
ax.set_xticklabels([f"Step {i + 1}" for i in range(6)])
ax.set_ylabel("Time per action (seconds)")
ax.set_xlabel("Action index")
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right")
ax.set_title(f"Training vs test: time per action (Disjunctive, n={n})", fontsize=12)

plt.tight_layout()
out = os.path.join(script_dir, "training_test_timing.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
