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
