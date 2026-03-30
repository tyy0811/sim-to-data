"""Evaluate conformal selective prediction across regimes.

Loads existing result JSONs (which contain softmax outputs) and
calibrates/evaluates conformal prediction on each regime.

Usage:
    python experiments/run_conformal.py
    python experiments/run_conformal.py --alpha 0.01
"""

import argparse
import json
import os

import numpy as np

from simtodata.evaluation.conformal import ConformalClassifier


def _load_result(path):
    """Load a result JSON and return (softmax, labels) arrays."""
    with open(path) as f:
        data = json.load(f)
    probs = np.array(data["y_proba"])
    labels = np.array(data["y_true"])
    return probs, labels


def _save_result(name, result, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"name": name, **result}, f, indent=2, default=str)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Conformal selective prediction")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Miscoverage rate (0.05 = 95%% coverage)")
    args = parser.parse_args()

    regimes = {
        "B1_source": "B1_cnn1d_source_on_source.json",
        "B2_shifted": "B2_cnn1d_source_on_shifted.json",
        "B5_shifted": "B5_cnn1d_randomized_finetune_on_shifted.json",
    }

    print(f"Conformal selective prediction (alpha={args.alpha})")
    print("=" * 60)

    all_results = {}
    for label, filename in regimes.items():
        path = os.path.join(args.results_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {label}: {path} not found")
            continue

        probs, labels = _load_result(path)

        # 50/50 calibration/evaluation split
        n_cal = len(labels) // 2
        cal_probs, eval_probs = probs[:n_cal], probs[n_cal:]
        cal_labels, eval_labels = labels[:n_cal], labels[n_cal:]

        cc = ConformalClassifier(alpha=args.alpha)
        cc.calibrate(cal_probs, cal_labels)
        result = cc.evaluate(eval_probs, eval_labels)

        print(f"\n  {label}:")
        print(f"    Coverage:        {result['coverage']:.3f}")
        print(f"    Abstention rate: {result['abstention_rate']:.3f}")
        print(f"    Effective F1:    {result['effective_f1']:.3f}")
        print(f"    q_hat:           {result['q_hat']:.4f}")
        print(f"    Per-class abstention: {result['class_abstention_rates']}")

        all_results[label] = result

    _save_result(
        "conformal_evaluation",
        {"alpha": args.alpha, "regimes": all_results},
        os.path.join(args.results_dir, "v3"),
    )

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Regime':<15} {'Coverage':>10} {'Abstain%':>10} {'Eff. F1':>10}")
    print("-" * 45)
    for label, r in all_results.items():
        print(f"{label:<15} {r['coverage']:>10.3f} {r['abstention_rate']*100:>9.1f}% "
              f"{r['effective_f1']:>10.3f}")


if __name__ == "__main__":
    main()
