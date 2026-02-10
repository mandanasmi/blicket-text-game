"""
Figure 3-style: 2x2 scatter plots of All Correct Accuracy vs.
(1) Number Unique States Visited, (2) Trial Information Gain,
(3) Average Response Time, (4) Number of Actions/Tests Before Q&A.
Uses round7 data.json. Light blue points, red LOWESS curve, shaded CI, Spearman rho + p.
"""

import argparse
import json
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = "sans-serif"


def _spearman_r_p(x, y):
    """Spearman rank correlation and approximate two-tailed p-value."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(np.sum(ok))
    if n < 3:
        return np.nan, np.nan
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    mx, my = np.mean(rx), np.mean(ry)
    sx = np.sqrt(np.mean((rx - mx) ** 2))
    sy = np.sqrt(np.mean((ry - my) ** 2))
    if sx == 0 or sy == 0:
        return np.nan, np.nan
    r = np.mean((rx - mx) * (ry - my)) / (sx * sy)
    r = float(np.clip(r, -1, 1))
    denom = 1 - r * r
    t = r * np.sqrt((n - 2) / denom) if denom > 1e-12 else (0.0 if abs(r) < 0.999 else 1e3)
    # approximate two-tailed p via normal (rough for small n)
    try:
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    except Exception:
        p = 0.05 if abs(t) > 2 else 0.5
    return float(r), float(max(1e-20, min(1, p)))


mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]
NUM_OBJECTS = 4

try:
    import statsmodels.api as sm
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False


def _get_true_blickets(rd):
    out = rd.get("true_blicket_indices")
    if out is not None:
        return out
    return (rd.get("config") or {}).get("blicket_indices")


def _get_true_rule(rd):
    out = rd.get("true_rule") or (rd.get("config") or {}).get("rule") or rd.get("rule")
    return (out or "").strip().lower()


def _parse_rule(s):
    if not s or not isinstance(s, str):
        return None
    s = s.lower()
    if "conjunctive" in s:
        return "conjunctive"
    if "disjunctive" in s or "any" in s:
        return "disjunctive"
    return None


def _get_chosen_blickets(rd):
    if "user_chosen_blickets" in rd:
        chosen = [x for x in (rd["user_chosen_blickets"] or []) if x is not None]
        return sorted(chosen) if chosen else None
    c = rd.get("blicket_classifications") or {}
    if not isinstance(c, dict):
        return None
    chosen = []
    for k, v in c.items():
        if v == "Yes":
            try:
                chosen.append(int(k.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return sorted(chosen) if chosen else None


def _has_prior(entry):
    comp = entry.get("comprehension") or {}
    sg = (comp if isinstance(comp, dict) else {}).get("similar_game_experience") or {}
    if not isinstance(sg, dict):
        return None
    a = (sg.get("answer") or "").strip().lower()
    if "yes" in a:
        return True
    if "no" in a:
        return False
    return None


def _all_subsets(n):
    out = []
    for b in range(1, 1 << n):
        out.append(frozenset(i for i in range(n) if (b >> i) & 1))
    return out


def _consistent(s, objects_on, machine_lit, conj):
    o = set(objects_on)
    pred = (s <= o) if conj else bool(s & o)
    return pred == machine_lit


def _least_dense_corner(x, y, frac=0.22):
    """Return (ax_x, ax_y), ha, va for placing text in the corner with fewest points."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 2:
        return 0.05, 0.95, "left", "top"
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
    candidates = [
        (x <= xmin + frac * xspan) & (y >= ymax - frac * yspan),
        (x >= xmax - frac * xspan) & (y >= ymax - frac * yspan),
        (x <= xmin + frac * xspan) & (y <= ymin + frac * yspan),
        (x >= xmax - frac * xspan) & (y <= ymin + frac * yspan),
    ]
    corners = [
        (0.05, 0.95, "left", "top"),
        (0.95, 0.95, "right", "top"),
        (0.05, 0.05, "left", "bottom"),
        (0.95, 0.05, "right", "bottom"),
    ]
    counts = [np.sum(c) for c in candidates]
    idx = int(np.argmin(counts))
    return corners[idx]


def _unique_states(state_history):
    seen = set()
    for ent in state_history:
        ob = ent.get("objects_on_machine")
        lit = ent.get("machine_lit")
        if ob is None or lit is None:
            continue
        o = tuple(sorted(ob)) if isinstance(ob, (list, tuple)) else ()
        seen.add((o, bool(lit)))
    return len(seen)


def _mean_info_gain(state_history, conj):
    hyps = _all_subsets(NUM_OBJECTS)
    remaining = set(hyps)
    igs = []
    for ent in state_history:
        ob = ent.get("objects_on_machine")
        lit = ent.get("machine_lit")
        if ob is None or lit is None:
            continue
        o = list(ob) if isinstance(ob, (list, tuple)) else []
        n_before = len(remaining)
        remaining = {s for s in remaining if _consistent(s, o, bool(lit), conj)}
        n_after = len(remaining)
        if n_before > 0 and n_after > 0:
            ig = math.log2(n_before / n_after)
            igs.append(ig)
    return float(np.mean(igs)) if igs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../human_active_data_no_prior_experience.json", help="Input JSON (default: from round7)")
    ap.add_argument("--output", default="accuracy_correlation_scatters.png")
    ap.add_argument("--no-prior-only", action="store_true")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    rows = []
    for _pid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if not args.no_prior_only and _has_prior(entry) is True:
            continue
        mg = entry.get("main_game") or {}
        if not isinstance(mg, dict):
            continue

        has_rounds = any(isinstance(k, str) and k.startswith("round_") for k in mg)
        rounds = (
            [(k, mg[k]) for k in mg if isinstance(k, str) and k.startswith("round_") and isinstance(mg.get(k), dict)]
            if has_rounds
            else [("main", mg)]
        )

        for _rk, rd in rounds:
            rule = _get_true_rule(rd)
            true_b = _get_true_blickets(rd)
            user_b = _get_chosen_blickets(rd)
            ur = _parse_rule(rd.get("rule_type") or "")
            rule_ok = ur and rule and rule == ur
            rule_01 = 1.0 if rule_ok else 0.0
            # Partial object accuracy: correct blickets / total true blickets
            if true_b is not None and len(true_b) > 0:
                true_set = set(true_b)
                user_set = set(user_b) if user_b is not None else set()
                n_correct = len(true_set & user_set)
                obj_partial = n_correct / len(true_set)
            else:
                obj_partial = np.nan
            # Combined partial accuracy: (objects + rule) / 2
            if np.isfinite(obj_partial):
                accuracy = (obj_partial + rule_01) / 2.0
            else:
                accuracy = np.nan

            sh = rd.get("state_history") or []
            if not isinstance(sh, list) or len(sh) < 1:
                continue
            tt = rd.get("test_timings") or []
            num_tests = len(tt) if isinstance(tt, list) else 0
            total_time = rd.get("total_test_time_seconds")
            if total_time is None and isinstance(tt, list) and tt:
                total_time = sum(
                    float(t.get("time_since_previous_seconds") or 0)
                    for t in tt if isinstance(t, dict)
                )
            if total_time is None:
                total_time = np.nan
            try:
                total_time = float(total_time)
            except Exception:
                total_time = np.nan

            n_states = _unique_states(sh)
            conj = rule == "conjunctive"
            mean_ig = _mean_info_gain(sh, conj)
            avg_resp = total_time / num_tests if num_tests and np.isfinite(total_time) else np.nan

            rows.append({
                "accuracy": accuracy,
                "num_unique_states": n_states,
                "mean_info_gain": mean_ig,
                "avg_response_time": avg_resp,
                "num_tests": num_tests,
                "rule": rule or "",
            })

    df = np.rec.fromrecords(
        [(r["accuracy"], r["num_unique_states"], r["mean_info_gain"], r["avg_response_time"], r["num_tests"], r["rule"]) for r in rows],
        names=["accuracy", "num_unique_states", "mean_info_gain", "avg_response_time", "num_tests", "rule"],
    )

    def spearman(x, y):
        return _spearman_r_p(x, y)

    def lowess_ci(x, y, frac=0.4, n_boot=80, grid_n=80):
        ok = np.isfinite(x) & np.isfinite(y)
        xc, yc = x[ok], y[ok]
        n = len(xc)
        if n < 4:
            return (None,) * 6
        order = np.argsort(xc)
        xu = np.linspace(xc.min(), xc.max(), grid_n)
        ys = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            xb, yb = xc[idx], yc[idx]
            if HAS_LOWESS:
                smooth = sm.nonparametric.lowess(yb, xb, frac=frac, return_sorted=True)
                yi = np.interp(xu, smooth[:, 0], smooth[:, 1])
            else:
                p = np.polyfit(xb, yb, 2)
                yi = np.polyval(p, xu)
            ys.append(yi)
        ys = np.array(ys)
        lo = np.nanpercentile(ys, 2.5, axis=0)
        hi = np.nanpercentile(ys, 97.5, axis=0)
        if HAS_LOWESS:
            smooth = sm.nonparametric.lowess(yc, xc, frac=frac, return_sorted=True)
            ym = np.interp(xu, smooth[:, 0], smooth[:, 1])
        else:
            p = np.polyfit(xc, yc, 2)
            ym = np.polyval(p, xu)
        return xc, yc, xu, ym, lo, hi

    panels = [
        ("num_unique_states", "States Visitations vs Accuracy", "Number Unique States Visited"),
        ("mean_info_gain", "Info Gain vs Accuracy", "Trial Information Gain"),
        ("avg_response_time", "Avg Response Time vs Accuracy", "Average Response Time (s)"),
        ("num_tests", "Num Steps Taken vs Accuracy", "Number of Actions/Tests Before Q&A"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.flatten()

    for ax, (xkey, title, xlabel) in zip(axes, panels):
        x = df[xkey]
        y = df["accuracy"]
        r, p = spearman(x, y)

        conj = (np.asarray(df["rule"]) == "conjunctive")
        disj = (np.asarray(df["rule"]) == "disjunctive")
        if np.any(conj):
            ax.scatter(x[conj], y[conj], c="#e67e22", alpha=0.6, s=40, edgecolors="none", label="Conjunctive")
        if np.any(disj):
            ax.scatter(x[disj], y[disj], c="#3498db", alpha=0.6, s=40, edgecolors="none", label="Disjunctive")

        res = lowess_ci(x, y)
        if res[0] is not None:
            xc, yc, xu, ym, lo, hi = res
            ax.plot(xu, ym, color="#c0392b", linewidth=2, zorder=2)
            ax.fill_between(xu, lo, hi, color="#c0392b", alpha=0.2, zorder=1)

        pstr = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        ax_x, ax_y, ha, va = _least_dense_corner(x, y)
        ax.text(ax_x, ax_y, f"Spearman's $\\rho$ = {r:.2f}\np = {pstr}", transform=ax.transAxes,
                fontsize=8, va=va, ha=ha,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Partial accuracy (objects + rule)" if ax in (axes[0], axes[2]) else "")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Correlation analysis",
                 fontsize=11, y=1.01)
    plt.tight_layout(pad=0.25)
    plt.subplots_adjust(hspace=0.32, wspace=0.28)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {args.output} (n={len(rows)})")


if __name__ == "__main__":
    main()
