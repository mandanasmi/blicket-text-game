"""
Correlate timing with all-correct Nexiom judgments in the disjunctive case.

Ground truth (from passive_app/disjunctive_case_app.py TEST_STEPS): only Object 6
is a Nexiom — answers must be No / No / Yes for objects 4, 5, 6.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "new-data-history.json")


def all_correct(answers: dict) -> bool:
    return (
        answers.get("object_4") == "No"
        and answers.get("object_5") == "No"
        and answers.get("object_6") == "Yes"
    )


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_r_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 8000, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) < 1e-12 or np.std(yi) < 1e-12:
            continue
        rs.append(np.corrcoef(xi, yi)[0, 1])
    rs = np.array(rs)
    r_obs = pearson_r(x, y)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r_obs, float(lo), float(hi)


def main() -> None:
    with open(data_path) as f:
        data = json.load(f)

    mean_train = []
    mean_test = []
    sum_read = []
    q_rt = []
    total_rt = []
    correct = []

    for record in data.values():
        if record.get("status") != "completed":
            continue
        d = record.get("disjunctive_case_data", {})
        a = d.get("nexiom_object_answers", {})
        tr = d.get("time_per_training_step_seconds")
        te = d.get("time_per_test_step_seconds")
        if not a or not tr or not te or len(tr) != 6 or len(te) != 6:
            continue
        tr = [float(x) for x in tr]
        te = [float(x) for x in te]
        mean_train.append(np.mean(tr))
        mean_test.append(np.mean(te))
        sum_read.append(sum(tr) + sum(te))
        q_rt.append(float(d["questions_response_time_seconds"]))
        total_rt.append(float(d["response_time_seconds"]))
        correct.append(1.0 if all_correct(a) else 0.0)

    y = np.array(correct)
    n = len(y)
    n_corr = int(y.sum())
    print(f"n = {n}  all-correct = {n_corr}  not = {n - n_corr}")
    print("Point-biserial correlation (Pearson r vs binary correct):\n")

    metrics = [
        ("Mean time per action (training)", np.array(mean_train)),
        ("Mean time per action (test)", np.array(mean_test)),
        ("Sum read time (train + test)", np.array(sum_read)),
        ("Questions response time", np.array(q_rt)),
        ("Total task time (response_time_seconds)", np.array(total_rt)),
    ]
    for label, x in metrics:
        r0, lo, hi = bootstrap_r_ci(x, y)
        print(f"  {label}")
        print(f"    r = {r0:+.3f}   95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]")
        print()

    # Figure: reading time and total task time by correctness
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    rng = np.random.default_rng(1)
    for ax, values, title in [
        (axes[0], sum_read, "Sum reading time (s)"),
        (axes[1], total_rt, "Total task time (s)"),
    ]:
        v0 = [v for v, c in zip(values, correct) if c == 0]
        v1 = [v for v, c in zip(values, correct) if c == 1]
        bp = ax.boxplot(
            [v1, v0],
            tick_labels=[
                "All correct\n(n={})".format(n_corr),
                "Not all correct\n(n={})".format(n - n_corr),
            ],
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], ["#5DCAA2", "#E8A598"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        j1 = rng.uniform(-0.08, 0.08, size=len(v1))
        j0 = rng.uniform(-0.08, 0.08, size=len(v0))
        ax.scatter(1 + j1, v1, color="#0F6E56", s=22, alpha=0.55, zorder=3)
        ax.scatter(2 + j0, v0, color="#8B3A2E", s=22, alpha=0.55, zorder=3)
        ax.set_ylabel(title)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Disjunctive: time vs all-correct (only Obj 6 is Nexiom)",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(script_dir, "time_vs_correctness.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
