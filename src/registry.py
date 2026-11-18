"""Model registry and the promotion gate.

The registry is a JSON file, not a service, but it enforces the thing a registry
exists to enforce: a model does not reach production because someone liked its
AUC.  It reaches production because it cleared every gate, and if it did not,
the registry records which gate stopped it and the incumbent stays live.

Two kinds of rule run here, and the distinction matters:

  * ABSOLUTE FLOORS -- a model that fails one of these is not fit to deploy at
    all.  The discrimination floor is not a number chosen by taste: it is
    measured at runtime from the best single-feature rule on the same data, so
    the gate asks "does this model beat the obvious thing it replaces?"

  * CHALLENGER RULES -- a model that clears the floors still does not displace
    the incumbent champion unless it is materially better and no worse on the
    things that are easy to trade away for AUC: calibration and segment parity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "registry.json"

# How much better a challenger has to be before it is worth the switch, and how
# much it is allowed to give back elsewhere while doing it.
MIN_AUC_GAIN = 0.005
MAX_ECE_REGRESSION = 0.005
MAX_GAP_REGRESSION = 0.02


@dataclass(frozen=True)
class Gate:
    """One absolute floor.

    `higher_is_better=False` means the metric must come in AT OR BELOW
    `threshold` (calibration error, latency), otherwise at or above it.
    """
    name: str
    metric: str
    threshold: float
    higher_is_better: bool = True
    rationale: str = ""

    def check(self, metrics: dict, overrides: dict | None = None) -> dict:
        threshold = (overrides or {}).get(self.name, self.threshold)
        value = metrics.get(self.metric)
        if value is None:
            return {"gate": self.name, "metric": self.metric, "value": None,
                    "threshold": threshold, "passed": False,
                    "detail": "metric missing from the candidate's evaluation"}
        passed = (value >= threshold) if self.higher_is_better else (value <= threshold)
        return {"gate": self.name, "metric": self.metric,
                "value": round(float(value), 4), "threshold": round(float(threshold), 4),
                "passed": bool(passed), "detail": self.rationale}


# Each floor exists because of a specific way a credit model fails in production.
GATES = [
    Gate("discrimination", "auc", 0.0, True,
         "must beat the best single-feature rule by a clear margin, or the "
         "model is not earning the pipeline it needs"),
    Gate("calibration", "ece", 0.03, False,
         "limits and provisions are set off predicted probabilities, so the "
         "number has to mean what it says, not merely rank correctly"),
    Gate("segment_stability", "max_false_decline_gap", 0.15, False,
         "no age or education band may absorb a disproportionate share of "
         "false declines at the operating cutoff"),
    Gate("latency", "score_ms_per_1k", 50.0, False,
         "batch scoring has to finish inside its window"),
]


def evaluate_gates(metrics: dict, overrides: dict | None = None) -> tuple[bool, list[dict]]:
    results = [g.check(metrics, overrides) for g in GATES]
    return all(r["passed"] for r in results), results


def challenger_verdict(candidate: dict, champion: dict | None) -> dict:
    """Compare a floor-clearing candidate against the incumbent."""
    if champion is None:
        return {"applies": False, "promote": True,
                "reason": "no incumbent; this candidate becomes the baseline champion"}

    gain = candidate["auc"] - champion["auc"]
    ece_delta = candidate["ece"] - champion["ece"]
    gap_delta = candidate["max_false_decline_gap"] - champion["max_false_decline_gap"]

    reasons = []
    if gain < MIN_AUC_GAIN:
        reasons.append(f"AUC gain {gain:+.4f} is below the {MIN_AUC_GAIN} switch threshold")
    if ece_delta > MAX_ECE_REGRESSION:
        reasons.append(f"calibration regressed {ece_delta:+.4f}")
    if gap_delta > MAX_GAP_REGRESSION:
        reasons.append(f"segment false-decline gap widened {gap_delta:+.4f}")

    return {"applies": True, "promote": not reasons,
            "auc_gain": round(gain, 4), "ece_delta": round(ece_delta, 4),
            "gap_delta": round(gap_delta, 4),
            "reason": "; ".join(reasons) or
                      f"beats the champion by {gain:+.4f} AUC with no regression"}


def load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"champion": None, "history": []}


def record(model_name: str, metrics: dict, params: dict,
           overrides: dict | None = None) -> dict:
    """Evaluate a candidate, write it to the registry, promote only if it earned it."""
    cleared_floors, gate_results = evaluate_gates(metrics, overrides)
    reg = load()
    champion = reg.get("champion")

    verdict = (challenger_verdict(metrics, champion) if cleared_floors
               else {"applies": False, "promote": False,
                     "reason": "did not clear the absolute floors"})
    promoted = cleared_floors and verdict["promote"]

    blocked = [r["gate"] for r in gate_results if not r["passed"]]
    if cleared_floors and not promoted:
        blocked = ["challenger_rule"]

    version = len(reg["history"]) + 1
    entry = {
        "version": version,
        "model": model_name,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "metrics": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                    for k, v in metrics.items()},
        "gates": gate_results,
        "challenger": verdict,
        "promoted": promoted,
        "blocked_by": blocked,
    }
    reg["history"].append(entry)
    if promoted:
        reg["champion"] = {
            "version": version, "model": model_name,
            "auc": float(metrics["auc"]), "ece": float(metrics["ece"]),
            "max_false_decline_gap": float(metrics["max_false_decline_gap"]),
        }
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2))
    return entry


def reset() -> None:
    """Start the registry empty, so a pipeline run is reproducible."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps({"champion": None, "history": []}, indent=2))
