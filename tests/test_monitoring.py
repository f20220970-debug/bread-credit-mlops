"""Tests for the metric functions.

These are the functions every downstream number depends on, so they are tested
against cases with a known answer rather than against a previous run's output.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import monitoring as M  # noqa: E402


def test_psi_is_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert M.psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(size=20000)
    small = M.psi(base, rng.normal(loc=0.1, size=20000))
    large = M.psi(base, rng.normal(loc=1.0, size=20000))
    assert 0 <= small < large
    assert large > M.PSI_ACT


def test_psi_is_symmetric_in_sign_of_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(size=20000)
    up = M.psi(base, rng.normal(loc=0.5, size=20000))
    down = M.psi(base, rng.normal(loc=-0.5, size=20000))
    assert up == pytest.approx(down, rel=0.25)


def test_psi_handles_a_constant_feature():
    """A column with no spread has no distribution to shift; it must not blow up."""
    const = np.ones(1000)
    assert M.psi(const, const) == 0.0
    assert np.isfinite(M.psi(const, np.ones(1000) * 2))


def test_psi_ignores_nans():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    y = x.copy()
    y[:500] = np.nan
    assert M.psi(x, y) < M.PSI_INVESTIGATE


def test_ks_bounds():
    rng = np.random.default_rng(3)
    x = rng.normal(size=4000)
    assert M.ks(x, x) == pytest.approx(0.0, abs=1e-9)
    far = M.ks(x, rng.normal(loc=6, size=4000))
    assert 0.95 < far <= 1.0


def test_calibration_error_is_zero_for_a_perfect_forecaster():
    """If the predicted rate equals the realised rate in every bin, ECE is 0."""
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, size=200000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert M.expected_calibration_error(y, p) < 0.01


def test_calibration_error_catches_a_constant_overstatement():
    rng = np.random.default_rng(5)
    p_true = rng.uniform(0.05, 0.45, size=100000)
    y = (rng.uniform(size=p_true.size) < p_true).astype(int)
    inflated = np.clip(p_true + 0.2, 0, 1)
    assert M.expected_calibration_error(y, inflated) == pytest.approx(0.2, abs=0.02)


def test_segment_stability_marks_thin_segments_unreliable():
    rng = np.random.default_rng(6)
    n = 1000
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < 0.3).astype(int)
    seg = pd.Series(["big"] * (n - 50) + ["tiny"] * 50)
    out = M.segment_stability(y, p, seg, cutoff=0.8, min_n=200)
    assert out.set_index("segment").loc["big", "reliable"]
    assert not out.set_index("segment").loc["tiny", "reliable"]


def test_false_decline_gap_excludes_unreliable_segments():
    table = pd.DataFrame({
        "segment": ["a", "b", "tiny"],
        "false_decline_rate": [0.10, 0.14, 0.90],
        "reliable": [True, True, False],
    })
    # 0.90 belongs to a segment too thin to trust and must not set the gap.
    assert M.false_decline_gap(table) == pytest.approx(0.04)


def test_auc_of_a_perfect_ranker_is_one():
    y = np.array([0, 0, 1, 1])
    assert M.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
