"""Drift, calibration and segment-stability metrics.

Everything here is a pure function over arrays so it can be unit-tested without
a model or a pipeline run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# PSI convention used across the industry and in this project's alert rules.
PSI_INVESTIGATE = 0.10
PSI_ACT = 0.25


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a baseline and a comparison sample.

    Bin edges come from the EXPECTED sample's quantiles, which is the whole
    point: we ask how the new data falls into the bins the baseline defined.
    Empty bins are floored so the log stays finite.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:  # a near-constant feature has no distribution to shift
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, bins=edges)[0].astype(float) / expected.size
    a = np.histogram(actual, bins=edges)[0].astype(float) / actual.size
    floor = 1e-6
    e, a = np.maximum(e, floor), np.maximum(a, floor)
    return float(np.sum((a - e) * np.log(a / e)))


def ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, computed on the pooled grid."""
    e = np.sort(np.asarray(expected, dtype=float))
    a = np.sort(np.asarray(actual, dtype=float))
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if e.size == 0 or a.size == 0:
        return float("nan")
    grid = np.union1d(e, a)
    return float(np.max(np.abs(np.searchsorted(e, grid, "right") / e.size
                               - np.searchsorted(a, grid, "right") / a.size)))


def feature_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI and KS per feature, sorted worst first, with the alert band attached."""
    rows = []
    for col in baseline.columns:
        p = psi(baseline[col].to_numpy(), current[col].to_numpy())
        rows.append({
            "feature": col,
            "psi": round(p, 4),
            "ks": round(ks(baseline[col].to_numpy(), current[col].to_numpy()), 4),
            "band": ("act" if p >= PSI_ACT
                     else "investigate" if p >= PSI_INVESTIGATE
                     else "stable"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Mean |observed - predicted| over equal-count score bins."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return float(abs(y_true.mean() - p.mean()))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, edges.size - 2)
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        total += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(total)


MIN_SEGMENT_N = 200


def segment_stability(y_true: np.ndarray, p: np.ndarray,
                      segments: pd.Series, cutoff: float,
                      min_n: int = MIN_SEGMENT_N) -> pd.DataFrame:
    """Per-segment decline rate and error rates at one fixed score cutoff.

    This is the responsible-AI check: a model can hold its overall AUC while
    one segment quietly absorbs most of the false declines.  Reporting it is
    the point; the thresholds that would make it acceptable are a policy
    decision, not a modelling one, so none is asserted here.

    Segments thinner than `min_n` are reported but marked unreliable, because a
    rate computed on a few dozen accounts swings on single cases and would
    otherwise set the gate on noise.
    """
    flag = p >= cutoff
    rows = []
    for name, m in segments.groupby(segments).groups.items():
        m = segments.index.isin(m)
        yt, fl = y_true[m], flag[m]
        pos, neg = yt == 1, yt == 0
        rows.append({
            "segment": name,
            "n": int(m.sum()),
            "base_rate": round(float(yt.mean()), 4),
            "decline_rate": round(float(fl.mean()), 4),
            # FNR: real defaults the cutoff let through.
            "miss_rate": round(float((~fl[pos]).mean()) if pos.any() else float("nan"), 4),
            # FPR: good accounts the cutoff declined.
            "false_decline_rate": round(float(fl[neg].mean()) if neg.any() else float("nan"), 4),
            "reliable": bool(m.sum() >= min_n),
        })
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def false_decline_gap(*tables: pd.DataFrame) -> float:
    """Widest false-decline spread across the reliable segments of every table."""
    worst = 0.0
    for t in tables:
        r = t[t["reliable"]]
        if len(r) > 1:
            worst = max(worst, float(r["false_decline_rate"].max()
                                     - r["false_decline_rate"].min()))
    return worst


def auc(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y_true, p))
