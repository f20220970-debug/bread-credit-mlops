"""Tests for the promotion gate and the serving monitor.

The gate is the part of this repo that says no, so most of these tests check
that it says no for the right reason.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F     # noqa: E402
import registry as R     # noqa: E402
import serving as S      # noqa: E402

GOOD = {"auc": 0.75, "ece": 0.01, "max_false_decline_gap": 0.05,
        "score_ms_per_1k": 5.0}


def test_a_sound_candidate_clears_every_floor():
    passed, results = R.evaluate_gates(GOOD, {"discrimination": 0.65})
    assert passed
    assert all(r["passed"] for r in results)


@pytest.mark.parametrize("metric,value,gate", [
    ("auc", 0.60, "discrimination"),
    ("ece", 0.20, "calibration"),
    ("max_false_decline_gap", 0.40, "segment_stability"),
    ("score_ms_per_1k", 500.0, "latency"),
])
def test_each_floor_blocks_its_own_failure(metric, value, gate):
    passed, results = R.evaluate_gates({**GOOD, metric: value},
                                       {"discrimination": 0.65})
    assert not passed
    failed = [r["gate"] for r in results if not r["passed"]]
    assert failed == [gate]


def test_a_missing_metric_fails_closed():
    """An evaluation that never produced the metric must not pass by default."""
    metrics = {k: v for k, v in GOOD.items() if k != "ece"}
    passed, results = R.evaluate_gates(metrics, {"discrimination": 0.65})
    assert not passed
    assert any(r["gate"] == "calibration" and r["value"] is None for r in results)


def test_discrimination_floor_is_overridable_at_runtime():
    """The floor is measured from the data, so the override has to bind."""
    _, results = R.evaluate_gates({**GOOD, "auc": 0.70}, {"discrimination": 0.80})
    disc = [r for r in results if r["gate"] == "discrimination"][0]
    assert disc["threshold"] == 0.80 and not disc["passed"]


def test_first_candidate_becomes_champion():
    v = R.challenger_verdict(GOOD, None)
    assert v["promote"] and not v["applies"]


def test_marginal_gain_does_not_displace_the_champion():
    champ = {"auc": 0.750, "ece": 0.010, "max_false_decline_gap": 0.05}
    v = R.challenger_verdict({**GOOD, "auc": 0.7510}, champ)
    assert not v["promote"] and "switch threshold" in v["reason"]


def test_auc_gain_cannot_buy_a_calibration_regression():
    champ = {"auc": 0.700, "ece": 0.010, "max_false_decline_gap": 0.05}
    v = R.challenger_verdict({**GOOD, "auc": 0.760, "ece": 0.030}, champ)
    assert not v["promote"] and "calibration regressed" in v["reason"]


def test_auc_gain_cannot_buy_a_widening_segment_gap():
    champ = {"auc": 0.700, "ece": 0.010, "max_false_decline_gap": 0.05}
    v = R.challenger_verdict({**GOOD, "auc": 0.760, "max_false_decline_gap": 0.12},
                             champ)
    assert not v["promote"] and "segment false-decline gap" in v["reason"]


# --- serving monitor ---------------------------------------------------------

BASELINE = {"score_mean": 0.22, "score_mean_sd": 0.004, "null_rate": 0.007,
            "latency_p95": 5.0, "score_std": 0.15, "score_std_floor": 0.075}


def _rec(**kw):
    base = {"batch": 1, "n": 2500, "null_rate": 0.007, "score_mean": 0.22,
            "score_p95": 0.6, "score_std": 0.15, "max_input_psi": 0.01,
            "worst_input_feature": "age", "latency_ms_per_1k": 4.0, "fault": None}
    return {**base, **kw}


def test_a_healthy_batch_raises_nothing():
    assert S.monitor(_rec(), BASELINE) == []


def test_null_spike_alerts():
    rules = [a.rule for a in S.monitor(_rec(null_rate=0.09), BASELINE)]
    assert "null_rate" in rules


def test_score_shift_alerts():
    rules = [a.rule for a in S.monitor(_rec(score_mean=0.28), BASELINE)]
    assert "score_shift" in rules


def test_latency_spike_alerts():
    rules = [a.rule for a in S.monitor(_rec(latency_ms_per_1k=40.0), BASELINE)]
    assert "latency" in rules


def test_input_drift_alerts_and_is_absent_from_output_only_mode():
    rec = _rec(max_input_psi=1.4, worst_input_feature="pay_status_latest")
    assert "input_drift" in [a.rule for a in S.monitor(rec, BASELINE)]
    assert S.monitor(rec, BASELINE, include_input_psi=False) == []


def test_an_empty_column_is_treated_as_maximum_input_drift():
    ref = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    chunk = ref.copy()
    chunk["b"] = np.nan
    worst, name = S._input_psi(chunk, ref)
    assert name == "b" and worst > 1.0


def test_unknown_fault_name_is_rejected():
    with pytest.raises(ValueError):
        S.inject_fault(None, pd.DataFrame({"a": [1.0]}), "not_a_fault")


# --- features ----------------------------------------------------------------

def test_both_windows_produce_the_same_feature_set():
    df = pd.DataFrame({
        "LIMIT_BAL": [100000.0, 50000.0], "SEX": [1, 2], "EDUCATION": [1, 2],
        "MARRIAGE": [1, 2], "AGE": [30, 45],
        **{f"PAY_{i}": [0, 1] for i in [0, 2, 3, 4, 5, 6]},
        **{f"BILL_AMT{i}": [10000.0, 20000.0] for i in range(1, 7)},
        **{f"PAY_AMT{i}": [1000.0, 2000.0] for i in range(1, 7)},
    })
    early, late = F.build_window(df, "early"), F.build_window(df, "late")
    assert list(early.columns) == list(late.columns) == F.FEATURES
    assert len(early) == len(late) == len(df)


def test_ratios_do_not_divide_by_zero():
    df = pd.DataFrame({
        "LIMIT_BAL": [100000.0], "SEX": [1], "EDUCATION": [1], "MARRIAGE": [1],
        "AGE": [30],
        **{f"PAY_{i}": [0] for i in [0, 2, 3, 4, 5, 6]},
        **{f"BILL_AMT{i}": [0.0] for i in range(1, 7)},   # a zero bill every month
        **{f"PAY_AMT{i}": [0.0] for i in range(1, 7)},
    })
    out = F.build_window(df, "late")
    assert np.isfinite(out.to_numpy(dtype=float)[~np.isnan(out.to_numpy(dtype=float))]).all()


def test_unknown_window_is_rejected():
    with pytest.raises(ValueError):
        F.build_window(pd.DataFrame(), "midway")
