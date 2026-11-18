"""Batch scoring, serving logs, and the monitor that reads them.

Two things here are measured rather than asserted:

  * the scoring latency is timed with `perf_counter` around the real predict call;
  * the per-batch score statistics come from scoring real accounts.

One thing is deliberately synthetic and labelled as such: `inject_fault` breaks a
batch on purpose so the monitor can be tested against a known answer.  A monitor
nobody has ever seen fire is not a monitor.  Detection rates reported anywhere in
this repo are detection rates ON INJECTED FAULTS, never a claim about production.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring import psi

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "results" / "serving_log.jsonl"

FAULTS = ("schema_drop", "stuck_feature", "score_shift", "latency_spike")


@dataclass
class Alert:
    batch: int
    rule: str
    detail: str
    severity: str


def _input_psi(chunk: pd.DataFrame, reference: pd.DataFrame | None) -> tuple[float, str]:
    """Worst per-feature PSI of a scored batch against the training distribution.

    This is the input-side half of the monitor.  A score-level check cannot see
    a feature that has gone constant if the model simply leans on its
    correlates instead; the input side can.
    """
    if reference is None:
        return float("nan"), ""
    worst, name = 0.0, ""
    for col in chunk.columns:
        live = chunk[col].to_numpy()
        if np.all(np.isnan(live.astype(float))) and not reference[col].isna().all():
            # The column arrived empty.  There is no distribution left to
            # compare, which is the most severe input fault there is.
            v = 9.99
        else:
            v = psi(reference[col].to_numpy(), live)
        if np.isfinite(v) and v > worst:
            worst, name = v, col
    return round(float(worst), 4), name


def _record(batch: int, chunk: pd.DataFrame, p: np.ndarray, elapsed_ms: float,
            reference: pd.DataFrame | None, fault: str | None) -> dict:
    worst_psi, worst_feature = _input_psi(chunk, reference)
    return {
        "batch": batch,
        "n": int(len(chunk)),
        "null_rate": round(float(chunk.isna().to_numpy().mean()), 6),
        "score_mean": round(float(p.mean()), 6),
        "score_p95": round(float(np.quantile(p, 0.95)), 6),
        "score_std": round(float(p.std()), 6),
        "max_input_psi": worst_psi,
        "worst_input_feature": worst_feature,
        "latency_ms_per_1k": round(float(elapsed_ms / len(chunk) * 1000), 3),
        "fault": fault,
    }


def score_batches(model, X: pd.DataFrame, n_batches: int = 12, seed: int = 0,
                  reference: pd.DataFrame | None = None) -> list[dict]:
    """Score X in batches, timing each call, and return one log record per batch.

    Batches are a random partition of the accounts.  They are NOT days: the
    source panel has no calendar column, so a batch is a unit of work, not a
    unit of time.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    records = []
    for i, idx in enumerate(np.array_split(order, n_batches)):
        chunk = X.iloc[idx]
        t0 = time.perf_counter()
        p = model.predict_proba(chunk)[:, 1]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        records.append(_record(i, chunk, p, elapsed_ms, reference, None))
    return records


def inject_fault(model, X: pd.DataFrame, fault: str, seed: int = 0,
                 reference: pd.DataFrame | None = None) -> dict:
    """Break one batch in a known way and return its log record.

    Each fault mirrors a real failure mode:
      schema_drop   - an upstream join drops a column, so it arrives all-null
      stuck_feature - a feature job fails and serves yesterday's constant
      score_shift   - a preprocessing change moves the score distribution
      latency_spike - the feature store slows down
    """
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    rng = np.random.default_rng(seed)
    chunk = X.iloc[rng.permutation(len(X))[:2500]].copy()

    if fault == "schema_drop":
        chunk["utilisation"] = np.nan
    elif fault == "stuck_feature":
        chunk["pay_status_latest"] = 0.0
    elif fault == "score_shift":
        # A unit error: limits arriving in thousands instead of units.
        chunk["limit_bal"] = chunk["limit_bal"] / 1000.0

    t0 = time.perf_counter()
    p = model.predict_proba(chunk)[:, 1]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if fault == "latency_spike":
        elapsed_ms *= 8

    return _record(-1, chunk, p, elapsed_ms, reference, fault)


def baseline_from(records: list[dict]) -> dict:
    """Summarise healthy batches into the thresholds the monitor compares against."""
    clean = [r for r in records if r["fault"] is None]
    return {
        "score_mean": float(np.mean([r["score_mean"] for r in clean])),
        "score_mean_sd": float(np.std([r["score_mean"] for r in clean])),
        "null_rate": float(np.mean([r["null_rate"] for r in clean])),
        "latency_p95": float(np.quantile([r["latency_ms_per_1k"] for r in clean], 0.95)),
        "score_std": float(np.mean([r["score_std"] for r in clean])),
        # A healthy batch's spread varies little, so half the baseline spread is
        # a floor only a genuinely collapsed score distribution falls through.
        "score_std_floor": float(np.mean([r["score_std"] for r in clean])) * 0.5,
    }


def monitor(record: dict, baseline: dict, include_input_psi: bool = True) -> list[Alert]:
    """Apply the serving-log alert rules to one batch.

    Deliberately boring rules.  Each one is something an on-call engineer can
    read off the log and act on without rerunning the model.

    `include_input_psi=False` reproduces the first version of this monitor,
    which watched only the model's outputs.  That version is kept callable on
    purpose: it is what missed the stuck upstream feature, and the comparison
    is the argument for watching inputs as well as outputs.
    """
    alerts: list[Alert] = []

    if include_input_psi and np.isfinite(record.get("max_input_psi", float("nan"))):
        if record["max_input_psi"] >= 0.25:
            alerts.append(Alert(record["batch"], "input_drift",
                                f"input PSI {record['max_input_psi']:.3f} on "
                                f"'{record['worst_input_feature']}' against the "
                                "training distribution", "high"))

    if record["null_rate"] > max(baseline["null_rate"] * 3, 0.01):
        alerts.append(Alert(record["batch"], "null_rate",
                            f"null rate {record['null_rate']:.3f} against a "
                            f"baseline of {baseline['null_rate']:.3f}", "high"))

    sd = baseline["score_mean_sd"] or 1e-9
    z = abs(record["score_mean"] - baseline["score_mean"]) / sd
    if z > 4:
        alerts.append(Alert(record["batch"], "score_shift",
                            f"mean score {record['score_mean']:.4f} is {z:.1f} sd "
                            f"from the baseline mean {baseline['score_mean']:.4f}",
                            "high" if z > 8 else "medium"))

    if record["latency_ms_per_1k"] > baseline["latency_p95"] * 3:
        alerts.append(Alert(record["batch"], "latency",
                            f"{record['latency_ms_per_1k']:.1f} ms/1k against a "
                            f"p95 of {baseline['latency_p95']:.1f}", "medium"))

    if record["score_std"] < baseline.get("score_std_floor", 0.02):
        alerts.append(Alert(record["batch"], "score_collapse",
                            f"score spread {record['score_std']:.4f} has collapsed, "
                            "which usually means a feature is stuck", "high"))
    return alerts


def write_log(records: list[dict], path: Path = LOG_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path
