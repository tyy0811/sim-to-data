# V3 Design: Safety-Critical Deployment Readiness

> A defect-classification pipeline that knows when not to decide —
> preserving safety coverage under domain shift by abstaining when
> uncertain and quantifying the human-review tradeoff.

**Scope:** 4 working days, zero GPU cost (all work uses existing checkpoints)
**Test target:** ~185 tests (from 167)

---

## What V3 Adds

1. **Conformal selective prediction** — classify or abstain, with
   distribution-free coverage guarantee (APS nonconformity scores)
2. **Cost-sensitive analysis** — expected inspection cost per 1000
   scans under configurable asymmetric cost matrix
3. **CORAL adaptation baseline** — one lightweight domain adaptation
   comparison (Sun & Saenko, 2016)
4. **ONNX export** — deployable inference artifact with parity
   verification

### What V3 Does Not Add

- DANN / adversarial adaptation (too complex for the payoff)
- 2D B-scan improvements (the negative result is more valuable as-is)
- Wavelet/spectral features (doesn't change the narrative)

---

## Design Decisions

### D1: Single-model conformal, not ensemble

The conformal coverage guarantee P(Y in C(X)) >= 1 - alpha is a
property of the calibration procedure, not the base model. It holds
for any predictor, even a bad one. A better model produces tighter
prediction sets (lower abstention), not a stronger guarantee.

**Primary path:** Calibrate on the seed_42 B5 checkpoint's softmax
outputs. This is sufficient for the coverage guarantee and ships
on Day 1.

**Stretch goal (Day 3/4):** Re-run multiseed training (~30 min CPU
background) to get 5 checkpoints. If it lands, add a bonus comparison
row: ensemble+conformal vs single-model+conformal. This demonstrates
that conformal works on any base model but benefits from a better one.

### D2: APS scores, not 1-p(true)

Adaptive Prediction Sets (Romano, Sesia & Candes, 2020) produce
smaller prediction sets than the simpler 1-p(true_class) threshold
because they account for the full probability ranking. For a 3-class
problem where low-severity is the hard class, this gives meaningfully
tighter sets and lower abstention at the same coverage level.

### D3: CORAL feature extractor must NOT detach gradients

The `FeatureExtractor` hook stores activations for CORAL loss
computation during training. Unlike Grad-CAM hooks (which run under
`torch.no_grad()` during inference), CORAL needs live gradients to
propagate alignment loss back through the feature layers.

```python
# CORRECT: CORAL training path
def _hook(self, module, input, output):
    self.features = output  # no detach

# WRONG: would make CORAL a silent no-op
def _hook(self, module, input, output):
    self.features = output.detach()  # breaks gradient graph
```

A dedicated test (`test_coral_gradient_reaches_model`) verifies that
CORAL loss gradients reach model conv parameters through the hook.

### D4: Cost matrix values are illustrative

The cost framework is the contribution, not the specific values. Real
deployment costs come from the operator's risk assessment. The default
matrix uses relative units (missed high-severity = 500, human review
= 5, false alarm = 1) to show the framework's mechanics.

### D5: Trace length is 1024

All A-scan inputs are 1024 samples. ONNX export defaults to this
length with dynamic batch axis.

### D6: CORAL expected result is modest/null

CORAL aligns second-order feature statistics. If the domain shift is
primarily in the physics (wave propagation parameters) rather than
feature covariance, CORAL won't help much. The honest finding "CORAL
provides minimal benefit over randomization for this signal class"
is the point — it shows a standard adaptation baseline was evaluated.

---

## File Structure

New files only. All existing code untouched.

```
src/simtodata/
  evaluation/
    conformal.py            # ConformalClassifier + APS + abstention
    cost.py                 # CostMatrix + expected cost + coverage sweep
  adaptation/
    __init__.py
    coral.py                # coral_loss + FeatureExtractor + train_with_coral
  export/
    __init__.py
    onnx_export.py          # export_to_onnx + verify_onnx
    onnx_infer.py           # run_inference (batch ONNX inference)

experiments/
  run_conformal.py          # Calibrate + evaluate selective prediction
  run_cost_analysis.py      # Cost sweep + operating-point figure
  run_coral.py              # CORAL fine-tuning (single seed, 4 weights)
  generate_v3_figures.py    # All V3 figures

tests/
  test_conformal.py         # 6 tests
  test_cost_framework.py    # 5 tests
  test_coral.py             # 5 tests (including gradient-reaches-model)
  test_onnx_export.py       # 3 tests

configs/
  cost_matrix.yaml          # Illustrative failure costs
```

---

## Implementation

### Conformal Selective Prediction

`src/simtodata/evaluation/conformal.py`

Core function `_conformal_quantile` uses the exact order statistic
k = ceil((N+1)(1-alpha)) without interpolation. `np.quantile` uses
interpolation by default, which can violate the finite-sample
guarantee.

`ConformalClassifier` workflow:
1. `calibrate(cal_softmax, cal_labels)` — compute APS nonconformity
   scores, set threshold q_hat via `_conformal_quantile`
2. `predict_sets(test_softmax)` — walk classes by descending
   probability, accumulate until cumsum >= q_hat
3. `predict_with_abstention(test_softmax)` — set size 1 = classify,
   set size > 1 = abstain (flag for human review)
4. `evaluate(test_softmax, test_labels)` — coverage, abstention rate,
   effective F1, per-class abstention rates

APS score for sample i: sort classes by descending probability, walk
the sorted list accumulating probability, score = cumulative
probability when the true class is reached. High score = true class
ranked low = poor model confidence.

### Cost-Sensitive Analysis

`src/simtodata/evaluation/cost.py`

`CostMatrix` dataclass with `default_ndt()` factory and `from_yaml()`
loader. Default costs (relative units):
- Correct: 0
- False alarm (predict defect, true no-defect): 1
- Miss low-severity: 50
- Miss high-severity: 500
- Human review (abstention): 5

`compute_expected_cost()` sums per-sample costs. Returns total,
per-sample, per-1000, and breakdown (classification vs review).

`sweep_coverage_vs_cost()` sweeps alpha from 0.01 to 0.50, at each
point: calibrate conformal classifier, predict with abstention,
compute cost. Produces the key V3 figure data: expected cost vs
coverage target. Uses 50/50 cal/eval split of the input data.

### CORAL Adaptation

`src/simtodata/adaptation/coral.py`

`coral_loss(source_features, target_features)`:
  loss = (1 / 4d^2) * ||C_s - C_t||^2_F
where C_s, C_t are feature covariance matrices.

`FeatureExtractor`: forward hook on named layer, stores activations
WITHOUT detaching (see D3). Single hook, persistent for training loop.

`train_with_coral()`: fine-tune with combined loss =
CE(source) + lambda * CORAL(source_features, target_features).
Target loader cycles if shorter than source. Returns per-epoch
loss history.

Feature layer for DefectCNN1D: `features.16` (AdaptiveAvgPool1d
before classifier head). Features shape after flatten: (B, 512).

### ONNX Export

`src/simtodata/export/onnx_export.py`

`export_to_onnx()`: trace with dummy input (1, 1, 1024), dynamic
batch axis, opset 14. `verify_onnx()`: compare N random samples
between PyTorch and ONNX runtime, assert allclose(atol=1e-5).

`src/simtodata/export/onnx_infer.py`

`run_inference()`: load ONNX session, run batch, manual softmax
(numpy), return predictions + probabilities + latency timing.

New dependency: `onnxruntime>=1.15` added to pyproject.toml optional
deps (not required for core functionality).

---

## Tests (19 new)

### test_conformal.py (6 tests)

1. `test_quantile_exact_order_statistic` — verify ceiling formula,
   not interpolation; check k > N returns inf
2. `test_calibrate_sets_q_hat` — after calibration, q_hat set and
   positive
3. `test_coverage_guarantee_empirical` — on 2000 synthetic samples
   with ~70% confident model, coverage >= 1-alpha (with finite-sample
   slack)
4. `test_abstention_increases_with_coverage` — alpha=0.05 abstains
   more than alpha=0.10
5. `test_predict_with_abstention_output_format` — shapes, dtypes,
   abstained samples have prediction=-1
6. `test_uncalibrated_raises` — RuntimeError before calibrate()

### test_cost_framework.py (5 tests)

1. `test_all_correct_zero_cost` — perfect predictions = zero cost
2. `test_missed_high_severity_is_catastrophic` — cost = 500
3. `test_abstention_costs_review` — cost = review_cost only
4. `test_review_cheaper_than_missed_defect` — abstain < miss
5. `test_cost_per_1000_scaling` — 1000x cost_per_sample

### test_coral.py (5 tests)

1. `test_coral_loss_zero_for_same_features` — identical inputs = ~0
2. `test_coral_loss_positive_for_different` — shifted mean > 0
3. `test_coral_loss_output_shape` — scalar
4. `test_coral_loss_gradient_flows` — requires_grad input gets grad
5. `test_coral_gradient_reaches_model` — CORAL loss through hook
   propagates to model conv parameters (catches the detach bug)

### test_onnx_export.py (3 tests)

1. `test_export_creates_file` — .onnx file exists after export
2. `test_onnx_output_shape` — (batch, 3) output from onnxruntime
3. `test_onnx_matches_pytorch` — verify_onnx returns True

---

## Experiment Scripts

### run_conformal.py

Load B5 seed_42 checkpoint. Get softmax on source_test and
shifted_test. Split each 50/50 for calibration/evaluation. Calibrate
at alpha=0.05 (95% coverage). Report: coverage, abstention rate,
effective F1 per regime. Save results JSON.

### run_cost_analysis.py

Load B2 and B5 softmax outputs on shifted_test. Sweep alpha 0.01 to
0.50. At each point: conformal calibrate, predict, compute expected
cost. Generate figure: expected cost vs coverage target (B2 vs B5
curves). Find optimal operating point per model.

### run_coral.py

Load B3 checkpoint. Fine-tune with CORAL on shifted target features.
Sweep coral_weight in [0.1, 0.5, 1.0, 5.0]. Pick best by val F1.
Report B6 row. Single seed (42) sufficient — baseline comparison.

### generate_v3_figures.py

1. Expected cost vs coverage target (B2 vs B5)
2. Per-class abstention rates (grouped bar chart)
3. Optional: prediction set size distribution

---

## README Structure (Day 4)

New sections inserted after existing Key Findings:

1. **Selective Prediction and Coverage Guarantees** — Table A
   (coverage / abstention / effective F1 per regime), figure, one
   paragraph on the tradeoff
2. **Expected Inspection Cost** — Table B (cost per 1000 scans),
   figure, note that values are illustrative
3. **Domain Adaptation Baseline (CORAL)** — B6 row, honest
   interpretation
4. **Deployment Considerations** — conformal recalibration, abstention
   policy, ONNX latency, EMAT transferability note

Updates to existing sections:
- Opening tagline updated for V3 narrative
- Honest Scope adds V3 caveats
- Engineering updates test count, adds onnxruntime to deps
- Quick Start adds ONNX inference example

---

## Execution Schedule

### Day 1: Conformal selective prediction
- Implement conformal.py
- Write test_conformal.py (6 tests)
- Run experiments/run_conformal.py on B5 seed_42
- Populate Table A with real numbers
- **Gate:** coverage >= 95% per regime, abstention rates differ

### Day 2: Cost-sensitive analysis
- Implement cost.py
- Write test_cost_framework.py (5 tests)
- Write configs/cost_matrix.yaml
- Run experiments/run_cost_analysis.py
- Generate expected_cost_vs_coverage figure
- **Gate:** clear figure showing conformalized B5 < B2 expected cost

### Day 3: CORAL baseline + ONNX export
- Implement coral.py (with correct non-detaching hook)
- Write test_coral.py (5 tests)
- Run experiments/run_coral.py (single seed, 4 weights)
- Implement onnx_export.py + onnx_infer.py
- Write test_onnx_export.py (3 tests)
- Stretch: kick off multiseed training in background
- **Gate:** B6 row populated, ONNX parity verified, ~185 tests pass

### Day 4: README rewrite + polish
- Restructure README with new sections
- Write selective prediction, cost, CORAL, deployment sections
- Stretch: if multiseed landed, add ensemble+conformal comparison row
- Full test pass, lint clean
- **Gate:** clone-to-understanding in 3 minutes
