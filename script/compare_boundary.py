"""
Compare boundary evaluation results from two trained models.

Usage:
    python script/compare_boundary.py <baseline_json> <regularized_json>

Example:
    python script/compare_boundary.py \
        output/no_boundary/boundary_eval_test_30000.json \
        output/with_boundary/boundary_eval_test_30000.json
"""

import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(v):
    return f"{v:.4f}"


def delta(a, b):
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:.4f})"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    base = load(sys.argv[1])
    reg  = load(sys.argv[2])

    print(f"\n{'':30s} {'Baseline':>12}  {'+ Boundary':>12}  {'Delta':>10}")
    print("-" * 70)

    for key, label in [
        ("mean_precision", "Precision"),
        ("mean_recall",    "Recall"),
        ("mean_f1",        "F1 (BF score)"),
    ]:
        bv = base[key]
        rv = reg[key]
        print(f"  {label:28s} {fmt(bv):>12}  {fmt(rv):>12}  {delta(bv, rv):>10}")

    print()
    print(f"  Split      : {base['split']}")
    print(f"  Tolerance  : {base['tolerance_px']} px")
    print(f"  Iterations : baseline={base['iteration']}  regularized={reg['iteration']}")
    print()
