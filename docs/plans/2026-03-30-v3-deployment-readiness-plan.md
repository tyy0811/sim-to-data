# V3 Deployment Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add conformal selective prediction, cost-sensitive evaluation, CORAL adaptation baseline, and ONNX export to the ultrasonic inspection pipeline.

**Architecture:** Conformal prediction calibrates on held-out softmax outputs from the existing B5 checkpoint (seed_42) to provide distribution-free coverage guarantees. Cost analysis sweeps coverage targets against an asymmetric cost matrix. CORAL fine-tunes B3 with covariance alignment loss. ONNX export wraps the best model for deployable inference.

**Tech Stack:** numpy, scipy (existing), torch (existing), scikit-learn (existing), onnxruntime (new optional dep), pyyaml (existing).

**Important — saved checkpoints use 3-block architecture:**
```python
DefectCNN1D(channels=(32, 64, 128), kernels=(7, 5, 3), fc_hidden=64, pool_size=1)
```
This differs from the 4-block default in `configs/model_cnn1d.yaml`. All tasks that load checkpoints must use this architecture. Feature layer for CORAL: `features.12` (AdaptiveAvgPool1d). Classifier input dim: 128.

**Existing result JSONs** (in `results/`) contain `y_proba` (N, 3) softmax outputs and `y_true` (N,) labels for 3000 test samples each. The conformal and cost experiments load these directly — no re-inference needed.

---

## Task 1: Implement conformal selective prediction

**Files:**
- Create: `src/simtodata/evaluation/conformal.py`
- Create: `tests/test_conformal.py`

**Step 1: Write the tests**

Create `tests/test_conformal.py`:

```python
"""Tests for conformal selective prediction."""

import numpy as np
import pytest

from simtodata.evaluation.conformal import (
    ConformalClassifier,
    _conformal_quantile,
)


def _softmax(x):
    """Numerically stable softmax for test data generation."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class TestConformalQuantile:
    def test_exact_order_statistic_overflow(self):
        """k > N should return inf."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        q = _conformal_quantile(scores, alpha=0.1)
        # k = ceil(6 * 0.9) = ceil(5.4) = 6 > N=5 → inf
        assert q == float("inf")

    def test_exact_order_statistic_value(self):
        """k <= N should return the k-th sorted value."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        q = _conformal_quantile(scores, alpha=0.2)
        # k = ceil(6 * 0.8) = ceil(4.8) = 5, sorted[4] = 0.9
        assert q == 0.9


class TestConformalClassifier:
    def test_calibrate_sets_q_hat(self):
        np.random.seed(42)
        cc = ConformalClassifier(alpha=0.1)
        probs = _softmax(np.random.randn(100, 3))
        labels = np.random.randint(0, 3, size=100)
        q = cc.calibrate(probs, labels)
        assert q > 0
        assert cc.q_hat == q

    def test_coverage_guarantee_empirical(self):
        np.random.seed(42)
        N = 2000
        true = np.random.randint(0, 3, size=N)
        probs = np.eye(3)[true] * 0.7 + 0.1
        probs = probs / probs.sum(axis=1, keepdims=True)
        n_cal = N // 2
        cc = ConformalClassifier(alpha=0.1)
        cc.calibrate(probs[:n_cal], true[:n_cal])
        result = cc.evaluate(probs[n_cal:], true[n_cal:])
        assert result["coverage"] >= 0.89  # finite-sample slack

    def test_abstention_increases_with_coverage(self):
        np.random.seed(42)
        probs = _softmax(np.random.randn(500, 3))
        labels = np.random.randint(0, 3, size=500)
        cc90 = ConformalClassifier(alpha=0.10)
        cc95 = ConformalClassifier(alpha=0.05)
        cc90.calibrate(probs[:250], labels[:250])
        cc95.calibrate(probs[:250], labels[:250])
        r90 = cc90.evaluate(probs[250:], labels[250:])
        r95 = cc95.evaluate(probs[250:], labels[250:])
        assert r95["abstention_rate"] >= r90["abstention_rate"]

    def test_predict_with_abstention_output_format(self):
        cc = ConformalClassifier(alpha=0.1)
        cc.q_hat = 0.8
        cc.n_classes = 3
        probs = _softmax(np.random.randn(20, 3))
        preds, abstained = cc.predict_with_abstention(probs)
        assert preds.shape == (20,)
        assert abstained.shape == (20,)
        assert set(np.unique(abstained)).issubset({True, False})
        assert all(preds[abstained] == -1)

    def test_uncalibrated_raises(self):
        cc = ConformalClassifier(alpha=0.1)
        with pytest.raises(RuntimeError):
            cc.predict_sets(np.zeros((5, 3)))
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_conformal.py -v
```

Expected: FAIL — `simtodata.evaluation.conformal` does not exist.

**Step 3: Implement conformal.py**

Create `src/simtodata/evaluation/conformal.py`:

```python
"""Conformal prediction for multi-class defect classification.

Provides distribution-free coverage guarantees with selective prediction:
classify when confident, abstain when uncertain (flag for human review).

Uses Adaptive Prediction Sets (APS) nonconformity scores, which produce
smaller prediction sets than the simpler 1-p(true) threshold method.

Based on:
- Romano, Sesia & Candes (2020). Classification with Valid Adaptive
  Prediction Sets.
- Angelopoulos & Bates (2021). A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification.
"""

from __future__ import annotations

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Exact order statistic for finite-sample conformal guarantee.

    Uses k = ceil((N+1)(1-alpha)) without interpolation.
    np.quantile interpolates by default, violating the finite-sample bound.

    Args:
        scores: (N,) nonconformity scores from calibration set.
        alpha: miscoverage rate (0.05 = 95% coverage).

    Returns:
        Calibrated threshold q_hat.
    """
    N = len(scores)
    k = int(np.ceil((N + 1) * (1 - alpha)))
    sorted_scores = np.sort(scores)
    if k > N:
        return float("inf")
    return float(sorted_scores[k - 1])


class ConformalClassifier:
    """Conformal selective prediction for safety-critical classification.

    Workflow:
        1. calibrate(cal_softmax, cal_labels) — set threshold q_hat
        2. predict_with_abstention(test_softmax) — classify or abstain

    Prediction set has 1 class = classify.
    Prediction set has >1 class = abstain (flag for human review).

    Coverage guarantee: P(true class in prediction set) >= 1 - alpha,
    regardless of the data distribution or model quality.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = None
        self.n_classes = None

    def calibrate(self, cal_softmax: np.ndarray, cal_labels: np.ndarray) -> float:
        """Calibrate on held-out data using APS nonconformity scores.

        APS score: sort classes by descending probability, walk down
        accumulating probability, score = cumulative when true class
        is reached. High score = true class ranked low = poor confidence.

        Args:
            cal_softmax: (N, C) softmax probabilities.
            cal_labels: (N,) integer true class labels.

        Returns:
            q_hat: calibrated APS threshold.
        """
        N, C = cal_softmax.shape
        self.n_classes = C
        scores = np.empty(N)

        for i in range(N):
            sorted_indices = np.argsort(-cal_softmax[i])
            cumsum = 0.0
            for idx in sorted_indices:
                cumsum += cal_softmax[i, idx]
                if idx == cal_labels[i]:
                    scores[i] = cumsum
                    break

        self.q_hat = _conformal_quantile(scores, self.alpha)
        return self.q_hat

    def predict_sets(self, test_softmax: np.ndarray) -> list:
        """Return prediction sets for each sample.

        Walk classes by descending probability, accumulate until
        cumulative probability >= q_hat.

        Args:
            test_softmax: (B, C) softmax probabilities.

        Returns:
            List of sets, each containing class indices.
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() before predict_sets()")

        pred_sets = []
        for i in range(test_softmax.shape[0]):
            sorted_indices = np.argsort(-test_softmax[i])
            cumsum = 0.0
            pset = set()
            for idx in sorted_indices:
                cumsum += test_softmax[i, idx]
                pset.add(int(idx))
                if cumsum >= self.q_hat:
                    break
            pred_sets.append(pset)
        return pred_sets

    def predict_with_abstention(self, test_softmax: np.ndarray):
        """Classify or abstain based on prediction set size.

        Args:
            test_softmax: (B, C) softmax probabilities.

        Returns:
            predictions: (B,) predicted class, -1 where abstained.
            abstained: (B,) boolean mask, True where abstained.
        """
        pred_sets = self.predict_sets(test_softmax)
        B = test_softmax.shape[0]
        predictions = np.full(B, -1, dtype=int)
        abstained = np.zeros(B, dtype=bool)

        for i, pset in enumerate(pred_sets):
            if len(pset) == 1:
                predictions[i] = next(iter(pset))
            else:
                abstained[i] = True

        return predictions, abstained

    def evaluate(self, test_softmax: np.ndarray, test_labels: np.ndarray) -> dict:
        """Full evaluation: coverage, abstention, effective metrics.

        Args:
            test_softmax: (B, C) softmax probabilities.
            test_labels: (B,) true labels.

        Returns:
            Dict with coverage, abstention_rate, effective_f1,
            class_abstention_rates, n_evaluated, n_abstained.
        """
        from sklearn.metrics import f1_score

        pred_sets = self.predict_sets(test_softmax)
        predictions, abstained = self.predict_with_abstention(test_softmax)

        B = len(test_labels)
        n_abstained = int(abstained.sum())
        n_evaluated = B - n_abstained

        covered = sum(1 for i in range(B) if test_labels[i] in pred_sets[i])
        coverage = covered / B

        if n_evaluated > 0:
            mask = ~abstained
            effective_f1 = float(f1_score(
                test_labels[mask], predictions[mask],
                average="macro", zero_division=0.0,
            ))
        else:
            effective_f1 = float("nan")

        class_abstention = {}
        for c in range(self.n_classes):
            class_mask = test_labels == c
            if class_mask.sum() > 0:
                class_abstention[c] = float(abstained[class_mask].mean())

        return {
            "coverage": coverage,
            "abstention_rate": n_abstained / B,
            "effective_f1": effective_f1,
            "n_evaluated": n_evaluated,
            "n_abstained": n_abstained,
            "n_total": B,
            "class_abstention_rates": class_abstention,
            "q_hat": self.q_hat,
            "alpha": self.alpha,
        }
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_conformal.py -v
```

Expected: 6 PASSED.

**Step 5: Run full test suite — no regressions**

```bash
python -m pytest tests/test_conformal.py tests/test_interpretability.py tests/test_metrics.py tests/test_transforms.py tests/test_calibration.py tests/test_regime.py tests/test_constants.py tests/test_noise.py tests/test_features.py tests/test_baselines.py tests/test_model_cnn1d.py tests/test_model_cnn2d.py tests/test_train_predict.py tests/test_forward_model.py tests/test_pipeline_integration.py -q
```

Expected: All pass.

**Step 6: Lint**

```bash
python -m ruff check src/simtodata/evaluation/conformal.py tests/test_conformal.py
```

Expected: All checks passed.

**Step 7: Commit**

```bash
git add src/simtodata/evaluation/conformal.py tests/test_conformal.py
git commit -m "feat(v3): conformal selective prediction with APS scores and coverage guarantee"
```

---

## Task 2: Implement conformal experiment script

**Files:**
- Create: `experiments/run_conformal.py`

**Step 1: Implement the script**

Create `experiments/run_conformal.py`:

```python
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
```

**Step 2: Run the script**

```bash
python experiments/run_conformal.py
```

Expected: Table printed with coverage >= 95% for each regime. Save JSON to `results/v3/conformal_evaluation.json`.

**Step 3: Lint**

```bash
python -m ruff check experiments/run_conformal.py
```

**Step 4: Commit**

```bash
git add experiments/run_conformal.py
git commit -m "feat(v3): conformal evaluation script for B1/B2/B5 regimes"
```

---

## Task 3: Implement cost-sensitive analysis

**Files:**
- Create: `src/simtodata/evaluation/cost.py`
- Create: `tests/test_cost_framework.py`
- Create: `configs/cost_matrix.yaml`

**Step 1: Write the tests**

Create `tests/test_cost_framework.py`:

```python
"""Tests for cost-sensitive evaluation framework."""

import numpy as np

from simtodata.evaluation.cost import CostMatrix, compute_expected_cost


class TestCostMatrix:
    def test_default_ndt_has_all_keys(self):
        cm = CostMatrix.default_ndt()
        for true_cls in range(3):
            for pred_cls in range(3):
                assert (true_cls, pred_cls) in cm.costs


class TestComputeExpectedCost:
    def test_all_correct_zero_cost(self):
        labels = np.array([0, 1, 2, 0, 1])
        preds = np.array([0, 1, 2, 0, 1])
        abstained = np.zeros(5, dtype=bool)
        cm = CostMatrix.default_ndt()
        result = compute_expected_cost(labels, preds, abstained, cm)
        assert result["total_cost"] == 0.0

    def test_missed_high_severity_is_catastrophic(self):
        labels = np.array([2])
        preds = np.array([0])
        abstained = np.array([False])
        cm = CostMatrix.default_ndt()
        result = compute_expected_cost(labels, preds, abstained, cm)
        assert result["total_cost"] == 500.0

    def test_abstention_costs_review(self):
        labels = np.array([2])
        preds = np.array([-1])
        abstained = np.array([True])
        cm = CostMatrix.default_ndt()
        result = compute_expected_cost(labels, preds, abstained, cm)
        assert result["total_cost"] == cm.review_cost

    def test_review_cheaper_than_missed_defect(self):
        cm = CostMatrix.default_ndt()
        miss = compute_expected_cost(
            np.array([2]), np.array([0]), np.array([False]), cm,
        )
        abstain = compute_expected_cost(
            np.array([2]), np.array([-1]), np.array([True]), cm,
        )
        assert abstain["total_cost"] < miss["total_cost"]

    def test_cost_per_1000_scaling(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        preds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        abstained = np.zeros(10, dtype=bool)
        cm = CostMatrix.default_ndt()
        result = compute_expected_cost(labels, preds, abstained, cm)
        assert abs(result["cost_per_1000"] - result["cost_per_sample"] * 1000) < 1e-6
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cost_framework.py -v
```

Expected: FAIL — `simtodata.evaluation.cost` does not exist.

**Step 3: Implement cost.py**

Create `src/simtodata/evaluation/cost.py`:

```python
"""Cost-sensitive evaluation for safety-critical defect classification.

Computes expected inspection cost under a configurable asymmetric cost
matrix, enabling operating-point analysis: at what coverage/abstention
tradeoff is the total inspection cost minimized?

The cost values are illustrative. The framework is the contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml


@dataclass
class CostMatrix:
    """Asymmetric cost matrix for defect classification.

    Attributes:
        costs: dict mapping (true_class, predicted_class) to cost.
        review_cost: cost per abstained sample (human review).
        class_names: list of class name strings for display.
    """
    costs: dict
    review_cost: float
    class_names: list

    @classmethod
    def default_ndt(cls) -> CostMatrix:
        """Default cost matrix for 3-class NDT defect classification.

        Relative units: missed high-severity = 500, human review = 5,
        false alarm = 1.
        """
        costs = {
            (0, 0): 0,    (0, 1): 1,    (0, 2): 1,
            (1, 0): 50,   (1, 1): 0,    (1, 2): 0,
            (2, 0): 500,  (2, 1): 0,    (2, 2): 0,
        }
        return cls(
            costs=costs,
            review_cost=5,
            class_names=["no_defect", "low_severity", "high_severity"],
        )

    @classmethod
    def from_yaml(cls, path: str) -> CostMatrix:
        """Load cost matrix from YAML config."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        costs = {}
        for key, val in cfg["costs"].items():
            parts = [int(x) for x in key.split(",")]
            costs[tuple(parts)] = val
        return cls(
            costs=costs,
            review_cost=cfg["review_cost"],
            class_names=cfg.get("class_names", []),
        )


def compute_expected_cost(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    abstained: np.ndarray,
    cost_matrix: CostMatrix,
) -> dict:
    """Compute expected inspection cost.

    Args:
        true_labels: (N,) true class labels.
        predictions: (N,) predicted class, -1 where abstained.
        abstained: (N,) boolean mask.
        cost_matrix: CostMatrix instance.

    Returns:
        Dict with total_cost, cost_per_sample, cost_per_1000,
        cost_breakdown.
    """
    N = len(true_labels)
    total = 0.0
    breakdown = {"classification_cost": 0.0, "review_cost": 0.0}

    for i in range(N):
        if abstained[i]:
            total += cost_matrix.review_cost
            breakdown["review_cost"] += cost_matrix.review_cost
        else:
            key = (int(true_labels[i]), int(predictions[i]))
            c = cost_matrix.costs.get(key, 0.0)
            total += c
            breakdown["classification_cost"] += c

    return {
        "total_cost": total,
        "cost_per_sample": total / N,
        "cost_per_1000": total / N * 1000,
        "cost_breakdown": breakdown,
        "n_samples": N,
    }


def sweep_coverage_vs_cost(
    softmax_probs: np.ndarray,
    true_labels: np.ndarray,
    cost_matrix: CostMatrix,
    alphas: Optional[np.ndarray] = None,
) -> list:
    """Sweep coverage targets and compute cost at each.

    Splits input 50/50 into calibration and evaluation sets.

    Args:
        softmax_probs: (N, C) model softmax outputs.
        true_labels: (N,) true labels.
        cost_matrix: CostMatrix instance.
        alphas: miscoverage rates to sweep (default: 0.01 to 0.50).

    Returns:
        List of dicts per alpha with coverage, abstention_rate,
        effective_f1, cost_per_1000.
    """
    from .conformal import ConformalClassifier

    if alphas is None:
        alphas = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    N = len(true_labels)
    n_cal = N // 2
    cal_probs, eval_probs = softmax_probs[:n_cal], softmax_probs[n_cal:]
    cal_labels, eval_labels = true_labels[:n_cal], true_labels[n_cal:]

    results = []
    for alpha in alphas:
        cc = ConformalClassifier(alpha=float(alpha))
        cc.calibrate(cal_probs, cal_labels)
        eval_result = cc.evaluate(eval_probs, eval_labels)
        preds, abstained_mask = cc.predict_with_abstention(eval_probs)
        cost_result = compute_expected_cost(
            eval_labels, preds, abstained_mask, cost_matrix,
        )
        results.append({
            "alpha": float(alpha),
            "target_coverage": 1 - float(alpha),
            **eval_result,
            **cost_result,
        })

    return results
```

**Step 4: Create configs/cost_matrix.yaml**

```yaml
# Illustrative cost matrix for ultrasonic defect classification.
# Values are relative units per inspection decision.
# Real values would come from the operator's risk assessment.

class_names: ["no_defect", "low_severity", "high_severity"]

# Format: "true_class,predicted_class" -> cost
costs:
  "0,0": 0      # correct: no defect
  "0,1": 1      # false alarm: predict low, true no-defect
  "0,2": 1      # false alarm: predict high, true no-defect
  "1,0": 50     # missed low-severity defect
  "1,1": 0      # correct: low severity
  "1,2": 0      # over-severity (conservative, no cost)
  "2,0": 500    # missed high-severity defect (catastrophic)
  "2,1": 0      # under-severity (still detected, no extra cost)
  "2,2": 0      # correct: high severity

# Cost of human review per abstained sample
review_cost: 5
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_cost_framework.py -v
```

Expected: 6 PASSED.

**Step 6: Lint**

```bash
python -m ruff check src/simtodata/evaluation/cost.py tests/test_cost_framework.py
```

**Step 7: Commit**

```bash
git add src/simtodata/evaluation/cost.py tests/test_cost_framework.py configs/cost_matrix.yaml
git commit -m "feat(v3): cost-sensitive evaluation framework with asymmetric cost matrix"
```

---

## Task 4: Implement cost analysis experiment and figure

**Files:**
- Create: `experiments/run_cost_analysis.py`

**Step 1: Implement the script**

Create `experiments/run_cost_analysis.py`:

```python
"""Cost-sensitive analysis across coverage operating points.

Sweeps alpha from 0.01 to 0.50 and computes expected inspection cost
for B2 (source-only) and B5 (randomized+finetuned) on shifted data.

Usage:
    python experiments/run_cost_analysis.py
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simtodata.evaluation.cost import CostMatrix, sweep_coverage_vs_cost


def _load_result(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["y_proba"]), np.array(data["y_true"])


def plot_cost_vs_coverage(results_b2, results_b5, save_path):
    """Plot expected cost per 1000 inspections vs coverage target."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cov_b2 = [r["target_coverage"] for r in results_b2]
    cost_b2 = [r["cost_per_1000"] for r in results_b2]
    cov_b5 = [r["target_coverage"] for r in results_b5]
    cost_b5 = [r["cost_per_1000"] for r in results_b5]

    ax.plot(cov_b2, cost_b2, "o-", color="#dd8452", linewidth=2, label="B2 (source-only)")
    ax.plot(cov_b5, cost_b5, "s-", color="#55a868", linewidth=2, label="B5 (randomized+ft)")

    ax.set_xlabel("Coverage Target (1 - α)")
    ax.set_ylabel("Expected Cost per 1000 Inspections")
    ax.set_title("Coverage vs Inspection Cost Tradeoff")
    ax.legend()
    ax.set_xlim(0.48, 1.01)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Cost-sensitive analysis")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "v3"), exist_ok=True)

    cm = CostMatrix.default_ndt()
    alphas = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    b2_probs, b2_labels = _load_result(
        os.path.join(args.results_dir, "B2_cnn1d_source_on_shifted.json"),
    )
    b5_probs, b5_labels = _load_result(
        os.path.join(args.results_dir, "B5_cnn1d_randomized_finetune_on_shifted.json"),
    )

    print("Sweeping B2 (source-only on shifted)...")
    results_b2 = sweep_coverage_vs_cost(b2_probs, b2_labels, cm, alphas)

    print("Sweeping B5 (randomized+ft on shifted)...")
    results_b5 = sweep_coverage_vs_cost(b5_probs, b5_labels, cm, alphas)

    # Save results
    sweep_data = {"cost_matrix": "default_ndt", "B2": results_b2, "B5": results_b5}
    out_path = os.path.join(args.results_dir, "v3", "cost_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(sweep_data, f, indent=2, default=str)
    print(f"Saved: {out_path}")

    # Generate figure
    plot_cost_vs_coverage(
        results_b2, results_b5,
        os.path.join(args.output_dir, "expected_cost_vs_coverage.png"),
    )

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<8} {'α':>6} {'Coverage':>10} {'Abstain%':>10} "
          f"{'Eff.F1':>8} {'Cost/1k':>10}")
    print("-" * 52)
    for label, results in [("B2", results_b2), ("B5", results_b5)]:
        for r in results:
            print(f"{label:<8} {r['alpha']:>6.2f} {r['coverage']:>10.3f} "
                  f"{r['abstention_rate']*100:>9.1f}% {r['effective_f1']:>8.3f} "
                  f"{r['cost_per_1000']:>10.1f}")
        print()


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

```bash
python experiments/run_cost_analysis.py
```

Expected: Table printed, figure saved to `docs/figures/expected_cost_vs_coverage.png`, results to `results/v3/cost_sweep_results.json`.

**Step 3: Lint**

```bash
python -m ruff check experiments/run_cost_analysis.py
```

**Step 4: Commit**

```bash
git add experiments/run_cost_analysis.py
git commit -m "feat(v3): cost analysis sweep with expected-cost-vs-coverage figure"
```

---

## Task 5: Implement CORAL adaptation

**Files:**
- Create: `src/simtodata/adaptation/__init__.py`
- Create: `src/simtodata/adaptation/coral.py`
- Create: `tests/test_coral.py`

**Step 1: Write the tests**

Create `tests/test_coral.py`:

```python
"""Tests for CORAL domain adaptation."""

import torch

from simtodata.adaptation.coral import coral_loss, FeatureExtractor
from simtodata.models.cnn1d import DefectCNN1D


class TestCoralLoss:
    def test_zero_for_same_features(self):
        x = torch.randn(32, 64)
        loss = coral_loss(x, x.clone())
        assert loss.item() < 1e-6

    def test_positive_for_different(self):
        src = torch.randn(32, 64)
        tgt = torch.randn(32, 64) + 2.0
        loss = coral_loss(src, tgt)
        assert loss.item() > 0

    def test_output_is_scalar(self):
        loss = coral_loss(torch.randn(16, 32), torch.randn(16, 32))
        assert loss.dim() == 0

    def test_gradient_flows_through_inputs(self):
        src = torch.randn(16, 32, requires_grad=True)
        tgt = torch.randn(16, 32)
        loss = coral_loss(src, tgt)
        loss.backward()
        assert src.grad is not None
        assert torch.all(torch.isfinite(src.grad))


class TestFeatureExtractorGradient:
    def test_coral_gradient_reaches_model(self):
        """CORAL loss gradients must propagate to model parameters."""
        model = DefectCNN1D(
            channels=(32, 64, 128), kernels=(7, 5, 3), fc_hidden=64, pool_size=1,
        )
        model.train()
        extractor = FeatureExtractor(model, "features.12")

        x_src = torch.randn(8, 1, 1024)
        x_tgt = torch.randn(8, 1, 1024)

        _ = model(x_src)
        src_feats = extractor.get()
        _ = model(x_tgt)
        tgt_feats = extractor.get()

        loss = coral_loss(src_feats, tgt_feats)
        loss.backward()

        conv_param = list(model.parameters())[0]
        assert conv_param.grad is not None
        assert conv_param.grad.abs().sum() > 0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_coral.py -v
```

Expected: FAIL — `simtodata.adaptation.coral` does not exist.

**Step 3: Implement coral.py**

Create `src/simtodata/adaptation/__init__.py` (empty file).

Create `src/simtodata/adaptation/coral.py`:

```python
"""CORAL: Correlation Alignment for domain adaptation.

Aligns second-order feature statistics (covariance) between source
and target domains during fine-tuning.

Sun & Saenko (2016). Deep CORAL: Correlation Alignment for Deep
Domain Adaptation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def coral_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """Compute CORAL loss between source and target feature batches.

    CORAL loss = (1 / 4d^2) * ||C_s - C_t||^2_F

    Args:
        source_features: (N_s, D) features from source domain.
        target_features: (N_t, D) features from target domain.

    Returns:
        Scalar CORAL loss.
    """
    d = source_features.shape[1]

    src_centered = source_features - source_features.mean(dim=0, keepdim=True)
    tgt_centered = target_features - target_features.mean(dim=0, keepdim=True)

    n_s = source_features.shape[0]
    n_t = target_features.shape[0]
    cov_s = (src_centered.T @ src_centered) / max(n_s - 1, 1)
    cov_t = (tgt_centered.T @ tgt_centered) / max(n_t - 1, 1)

    diff = cov_s - cov_t
    return (diff * diff).sum() / (4 * d * d)


class FeatureExtractor:
    """Hook-based feature extractor for a named layer.

    Stores activations WITHOUT detaching — CORAL needs live gradients
    to propagate alignment loss back through the feature layers.

    Usage:
        extractor = FeatureExtractor(model, 'features.12')
        output = model(x)
        features = extractor.get()  # (B, D) with grad
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.features = None
        layer = dict(model.named_modules())[layer_name]
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.features = output  # no detach — CORAL needs live gradients

    def get(self) -> Optional[torch.Tensor]:
        return self.features

    def remove(self):
        """Remove the forward hook."""
        self._handle.remove()


def train_with_coral(
    model: nn.Module,
    source_loader,
    target_loader,
    optimizer: torch.optim.Optimizer,
    feature_layer: str,
    coral_weight: float = 1.0,
    n_epochs: int = 20,
    device: str = "cpu",
) -> list:
    """Fine-tune model with CORAL regularization.

    Total loss = CE(source) + coral_weight * CORAL(source_feats, target_feats).

    Args:
        model: pre-trained CNN.
        source_loader: labeled source domain data.
        target_loader: unlabeled target domain data.
        optimizer: optimizer for fine-tuning.
        feature_layer: name of layer for feature extraction.
        coral_weight: lambda for CORAL loss.
        n_epochs: fine-tuning epochs.
        device: torch device.

    Returns:
        List of per-epoch loss dicts.
    """
    extractor = FeatureExtractor(model, feature_layer)
    ce_loss_fn = nn.CrossEntropyLoss()
    history = []

    try:
        for epoch in range(n_epochs):
            model.train()
            epoch_ce, epoch_coral, n_batches = 0.0, 0.0, 0

            target_iter = iter(target_loader)
            for src_x, src_y in source_loader:
                try:
                    tgt_x, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    tgt_x, _ = next(target_iter)

                src_x, src_y = src_x.to(device), src_y.to(device)
                tgt_x = tgt_x.to(device)

                src_out = model(src_x)
                src_feats = extractor.get()
                ce = ce_loss_fn(src_out, src_y)

                _ = model(tgt_x)
                tgt_feats = extractor.get()

                c_loss = coral_loss(src_feats, tgt_feats)
                total = ce + coral_weight * c_loss

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                epoch_ce += ce.item()
                epoch_coral += c_loss.item()
                n_batches += 1

            history.append({
                "epoch": epoch,
                "ce_loss": epoch_ce / max(n_batches, 1),
                "coral_loss": epoch_coral / max(n_batches, 1),
            })
    finally:
        extractor.remove()

    return history
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_coral.py -v
```

Expected: 5 PASSED.

**Step 5: Lint**

```bash
python -m ruff check src/simtodata/adaptation/ tests/test_coral.py
```

**Step 6: Commit**

```bash
git add src/simtodata/adaptation/__init__.py src/simtodata/adaptation/coral.py tests/test_coral.py
git commit -m "feat(v3): CORAL domain adaptation with non-detaching feature hook"
```

---

## Task 6: Implement CORAL experiment script

**Files:**
- Create: `experiments/run_coral.py`

**Step 1: Implement the script**

Create `experiments/run_coral.py`:

```python
"""CORAL adaptation baseline experiment.

Fine-tunes B3 (randomized) with CORAL loss on shifted target features.
Sweeps coral_weight in [0.1, 0.5, 1.0, 5.0], picks best by val F1.

Usage:
    python experiments/run_coral.py
    python experiments/run_coral.py --quick
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from simtodata.adaptation.coral import train_with_coral
from simtodata.data.dataset import InspectionDataset
from simtodata.data.transforms import Normalize
from simtodata.evaluation.metrics import compute_all_metrics
from simtodata.models.cnn1d import DefectCNN1D
from simtodata.models.predict import predict_batch


def _seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seeded_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def main():
    parser = argparse.ArgumentParser(description="CORAL adaptation baseline")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    _seed_everything(args.seed)
    os.makedirs(os.path.join(args.results_dir, "v3"), exist_ok=True)

    norm = Normalize()
    source_train = InspectionDataset(f"{args.data_dir}/source_train.npz", transform=norm)
    shifted_test = InspectionDataset(f"{args.data_dir}/shifted_test.npz", transform=norm)
    adapt_data = InspectionDataset(f"{args.data_dir}/shifted_adapt.npz", transform=norm)

    bs = 128
    source_loader = _seeded_loader(source_train, bs, shuffle=True, seed=args.seed)
    target_loader = _seeded_loader(adapt_data, bs, shuffle=True, seed=args.seed)
    test_loader = _seeded_loader(shifted_test, bs, shuffle=False, seed=args.seed)

    weights = [0.1, 0.5, 1.0, 5.0]
    n_epochs = 5 if args.quick else 20

    best_f1, best_weight, best_metrics = 0.0, None, None
    all_results = []

    for w in weights:
        print(f"\nCORAL weight={w}:")
        _seed_everything(args.seed)

        model = DefectCNN1D(
            channels=(32, 64, 128), kernels=(7, 5, 3), fc_hidden=64, pool_size=1,
        )
        model.load_state_dict(
            torch.load(f"{args.models_dir}/B3_cnn1d_randomized.pt", weights_only=True),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

        history = train_with_coral(
            model, source_loader, target_loader, optimizer,
            feature_layer="features.12", coral_weight=w, n_epochs=n_epochs,
        )

        preds, probs, labels = predict_batch(model, test_loader)
        metrics = compute_all_metrics(labels, preds, probs)

        print(f"  Macro-F1: {metrics['macro_f1']:.4f}  AUROC: {metrics['auroc']:.4f}")
        all_results.append({"coral_weight": w, "metrics": metrics, "history": history})

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_weight = w
            best_metrics = metrics
            best_probs, best_labels, best_preds = probs, labels, preds

    print(f"\nBest: weight={best_weight}, F1={best_f1:.4f}")

    # Save best result as B6
    result = {"name": "B6_cnn1d_coral", "metrics": best_metrics}
    result["y_true"] = best_labels.tolist()
    result["y_pred"] = best_preds.tolist()
    result["y_proba"] = best_probs.tolist()
    result["coral_weight"] = best_weight
    out_path = os.path.join(args.results_dir, "v3", "B6_cnn1d_coral_on_shifted.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    # Save sweep summary
    sweep_path = os.path.join(args.results_dir, "v3", "coral_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump({"seed": args.seed, "results": all_results}, f, indent=2)
    print(f"Saved: {sweep_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script**

```bash
python experiments/run_coral.py --quick
```

Expected: Sweep 4 weights, print F1 for each, save B6 result JSON.

**Step 3: Lint**

```bash
python -m ruff check experiments/run_coral.py
```

**Step 4: Commit**

```bash
git add experiments/run_coral.py
git commit -m "feat(v3): CORAL adaptation experiment with weight sweep"
```

---

## Task 7: Implement ONNX export

**Files:**
- Create: `src/simtodata/export/__init__.py`
- Create: `src/simtodata/export/onnx_export.py`
- Create: `src/simtodata/export/onnx_infer.py`
- Create: `tests/test_onnx_export.py`

**Step 1: Write the tests**

Create `tests/test_onnx_export.py`:

```python
"""Tests for ONNX export and inference."""

import numpy as np
import torch

from simtodata.export.onnx_export import export_to_onnx, verify_onnx
from simtodata.models.cnn1d import DefectCNN1D


def _build_model():
    """Build the 3-block model matching saved checkpoints."""
    model = DefectCNN1D(
        channels=(32, 64, 128), kernels=(7, 5, 3), fc_hidden=64, pool_size=1,
    )
    model.eval()
    return model


class TestOnnxExport:
    def test_export_creates_file(self, tmp_path):
        model = _build_model()
        path = str(tmp_path / "test.onnx")
        export_to_onnx(model, path, trace_length=1024)
        assert (tmp_path / "test.onnx").exists()

    def test_onnx_output_shape(self, tmp_path):
        import onnxruntime as ort

        model = _build_model()
        path = str(tmp_path / "test.onnx")
        export_to_onnx(model, path, trace_length=1024)
        session = ort.InferenceSession(path)
        x = np.random.randn(4, 1, 1024).astype(np.float32)
        out = session.run(None, {"trace": x})[0]
        assert out.shape == (4, 3)

    def test_onnx_matches_pytorch(self, tmp_path):
        model = _build_model()
        path = str(tmp_path / "test.onnx")
        export_to_onnx(model, path, trace_length=1024)
        assert verify_onnx(model, path, trace_length=1024, atol=1e-5)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_onnx_export.py -v
```

Expected: FAIL — `simtodata.export.onnx_export` does not exist.

**Step 3: Implement onnx_export.py and onnx_infer.py**

Create `src/simtodata/export/__init__.py` (empty file).

Create `src/simtodata/export/onnx_export.py`:

```python
"""Export trained CNN to ONNX for deployable inference."""

from __future__ import annotations

import numpy as np
import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    trace_length: int = 1024,
    opset_version: int = 14,
) -> str:
    """Export PyTorch model to ONNX.

    Args:
        model: trained CNN in eval mode.
        output_path: where to save .onnx file.
        trace_length: input A-scan length.
        opset_version: ONNX opset.

    Returns:
        output_path.
    """
    model.eval()
    dummy_input = torch.randn(1, 1, trace_length)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["trace"],
        output_names=["logits"],
        dynamic_axes={
            "trace": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
    )
    return output_path


def verify_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    trace_length: int = 1024,
    n_samples: int = 10,
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX output matches PyTorch within tolerance.

    Args:
        model: PyTorch model in eval mode.
        onnx_path: path to exported ONNX model.
        trace_length: input length.
        n_samples: number of random samples to verify.
        atol: absolute tolerance for comparison.

    Returns:
        True if all samples match within tolerance.
    """
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(onnx_path)

    for _ in range(n_samples):
        x = torch.randn(1, 1, trace_length)
        with torch.no_grad():
            pt_out = model(x).numpy()
        onnx_out = session.run(None, {"trace": x.numpy()})[0]

        if not np.allclose(pt_out, onnx_out, atol=atol):
            return False

    return True
```

Create `src/simtodata/export/onnx_infer.py`:

```python
"""Batch inference with ONNX model."""

from __future__ import annotations

import time

import numpy as np


def run_inference(onnx_path: str, traces: np.ndarray) -> dict:
    """Run batch inference and return predictions with timing.

    Args:
        onnx_path: path to ONNX model.
        traces: (N, 1, L) input traces.

    Returns:
        Dict with predictions, probabilities, latency_ms.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)

    start = time.perf_counter()
    logits = session.run(None, {"trace": traces.astype(np.float32)})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return {
        "predictions": probs.argmax(axis=1).tolist(),
        "probabilities": probs.tolist(),
        "n_samples": len(traces),
        "latency_ms": round(elapsed_ms, 2),
        "latency_per_sample_ms": round(elapsed_ms / len(traces), 4),
    }
```

**Step 4: Add onnxruntime to optional deps**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
]
export = [
    "onnxruntime>=1.15",
]
```

**Step 5: Install onnxruntime and run tests**

```bash
pip install onnxruntime>=1.15
python -m pytest tests/test_onnx_export.py -v
```

Expected: 3 PASSED.

**Step 6: Lint**

```bash
python -m ruff check src/simtodata/export/ tests/test_onnx_export.py
```

**Step 7: Commit**

```bash
git add src/simtodata/export/__init__.py src/simtodata/export/onnx_export.py src/simtodata/export/onnx_infer.py tests/test_onnx_export.py pyproject.toml
git commit -m "feat(v3): ONNX export with parity verification and batch inference"
```

---

## Task 8: Run full experiments, generate all V3 figures

**Files:**
- Run: `experiments/run_conformal.py`
- Run: `experiments/run_cost_analysis.py`
- Run: `experiments/run_coral.py`

**Step 1: Run conformal experiment (if not already run in Task 2)**

```bash
python experiments/run_conformal.py
```

Record the output numbers for README.

**Step 2: Run cost analysis**

```bash
python experiments/run_cost_analysis.py
```

Record the output numbers and verify figure at `docs/figures/expected_cost_vs_coverage.png`.

**Step 3: Run CORAL experiment**

```bash
python experiments/run_coral.py
```

Record B6 results for README.

**Step 4: Export ONNX model**

```bash
python -c "
import torch
from simtodata.models.cnn1d import DefectCNN1D
from simtodata.export.onnx_export import export_to_onnx, verify_onnx

model = DefectCNN1D(channels=(32, 64, 128), kernels=(7, 5, 3), fc_hidden=64, pool_size=1)
model.load_state_dict(torch.load('models/B5_cnn1d_randomized_finetuned.pt', weights_only=True))
model.eval()
export_to_onnx(model, 'models/B5_cnn1d_randomized_finetuned.onnx')
assert verify_onnx(model, 'models/B5_cnn1d_randomized_finetuned.onnx')
print('ONNX export verified.')
"
```

**Step 5: Commit experiment outputs**

```bash
git add results/v3/ docs/figures/expected_cost_vs_coverage.png
git commit -m "feat(v3): experiment results — conformal, cost sweep, CORAL, ONNX"
```

---

## Task 9: Final verification

**Step 1: Run all safe tests**

```bash
python -m pytest tests/test_conformal.py tests/test_cost_framework.py tests/test_coral.py tests/test_onnx_export.py tests/test_interpretability.py tests/test_metrics.py tests/test_transforms.py tests/test_calibration.py tests/test_regime.py tests/test_constants.py tests/test_noise.py tests/test_features.py tests/test_baselines.py tests/test_model_cnn1d.py tests/test_model_cnn2d.py tests/test_train_predict.py tests/test_forward_model.py tests/test_pipeline_integration.py -q
```

Expected: ~185 tests pass.

**Step 2: Lint everything**

```bash
python -m ruff check src/ tests/ experiments/
```

Expected: All checks passed.

**Step 3: Commit any fixes**

Only if needed.

---

## Task 10: Update README for V3

**Files:**
- Modify: `README.md`

This task uses the real numbers from Task 8 experiments. Update README per the structure in the design doc:

1. Update opening tagline
2. Insert "Selective Prediction and Coverage Guarantees" section after Key Findings
3. Insert "Expected Inspection Cost" section
4. Insert "Domain Adaptation Baseline (CORAL)" section with B6 row
5. Insert "Deployment Considerations" section
6. Update Honest Scope for V3 caveats
7. Update Engineering section: test count, add onnxruntime dep note
8. Add ONNX inference example to Quick Start

Use actual numbers from the experiment JSONs in `results/v3/`.

**Commit:**

```bash
git add README.md
git commit -m "docs(v3): README with selective prediction, cost analysis, CORAL, deployment"
```

---

## Stretch: Ensemble conformal comparison (Day 3/4 if time allows)

**Not a blocking task.** Kick off multiseed training in background:

```bash
python experiments/run_multiseed.py &
```

If it completes (~30 min), save all 5 seed checkpoints, average softmax across 5 models, re-run conformal evaluation, and add a comparison row to the README showing ensemble+conformal vs single-model+conformal.
