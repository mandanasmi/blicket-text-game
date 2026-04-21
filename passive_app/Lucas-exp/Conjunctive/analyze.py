import json
import math
import sys

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_stats(values, binary=True):
    """
    For binary (0/1) data, uses the binomial SEM: sqrt(p*(1-p)/n).
    For count data, uses the sample SEM: std / sqrt(n).
    """
    n = len(values)
    mean = sum(values) / n
    if binary:
        sem = math.sqrt(mean * (1 - mean) / n)
    else:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        sem = math.sqrt(variance) / math.sqrt(n)
    return mean, sem, n

def main(filepath):
    data = load_data(filepath)

    o4, o5, o6 = [], [], []

    for pid, record in data.items():
        if record.get("status") != "completed":
            continue
        case = record.get("conjunctive_case_data")
        if not case:
            continue
        answers = case.get("nexiom_object_answers", {})
        o4.append(1 if answers.get("object_4") == "Yes" else 0)
        o5.append(1 if answers.get("object_5") == "Yes" else 0)
        o6.append(1 if answers.get("object_6") == "Yes" else 0)

    n = len(o4)
    print(f"Completed participants: {n}\n")
    print(f"{'Object':<12} {'Yes count':>10} {'Mean':>10} {'SEM':>10}")
    print("-" * 45)

    for label, vals in [("Object 4", o4), ("Object 5", o5), ("Object 6", o6)]:
        mean, sem, _ = compute_stats(vals, binary=True)
        print(f"{label:<12} {sum(vals):>10} {mean:>10.4f} {sem:>10.4f}")

    # Total yes count per participant across all 3 objects
    combined = [a + b + c for a, b, c in zip(o4, o5, o6)]
    mean_c, sem_c, _ = compute_stats(combined, binary=False)
    print(f"\nTotal 'Yes' per participant (objects 4+5+6 combined):")
    print(f"  Mean = {mean_c:.4f}, SEM = {sem_c:.4f}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data.json"
    main(filepath)