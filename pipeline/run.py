"""End-to-end pipeline: ingest -> features -> train -> evaluate -> gate -> serve -> monitor.

Run it with `python pipeline/run.py`.  Every stage writes its artifact to
results/ and prints a one-line status, so a failed run says which stage failed
rather than only that it failed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F           # noqa: E402
import monitoring as M         # noqa: E402
import registry as R           # noqa: E402
import serving as S            # noqa: E402

RESULTS = ROOT / "results"
SEED = 0
CUTOFF_RATE = 0.20   # decline the riskiest 20% -- the operating point everything is read at


def stage(name: str):
    print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)


def build_candidates():
    """The champion is the simple one.  A challenger has to earn its complexity."""
    return {
        "logistic_baseline": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=300,
            l2_regularization=1.0, random_state=SEED),
        # A candidate that ranks acceptably and is badly calibrated.  It is here
        # because a gate nobody has ever seen reject anything is decoration.
        "naive_bayes": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), GaussianNB()),
    }


def evaluate(model, Xtr, ytr, Xte, yte, X_late, y, age_band, edu_band) -> dict:
    model.fit(Xtr, ytr)

    t0 = time.perf_counter()
    p = model.predict_proba(Xte)[:, 1]
    per_row_ms = (time.perf_counter() - t0) / len(Xte) * 1000

    cutoff = float(np.quantile(p, 1 - CUTOFF_RATE))
    seg_age = M.segment_stability(yte.to_numpy(), p, age_band.loc[yte.index], cutoff)
    seg_edu = M.segment_stability(yte.to_numpy(), p, edu_band.loc[yte.index], cutoff)
    gap = M.false_decline_gap(seg_age, seg_edu)

    # The same fitted model scored on the LATER billing window, for every account.
    p_late = model.predict_proba(X_late)[:, 1]

    return {
        "metrics": {
            "auc": M.auc(yte, p),
            "ece": M.expected_calibration_error(yte.to_numpy(), p),
            "auc_late_window": M.auc(y, p_late),
            "max_false_decline_gap": gap,
            # p99 of a per-row cost measured over the batch call.
            "score_ms_per_1k": per_row_ms * 1000,
            "cutoff": cutoff,
        },
        "seg_age": seg_age,
        "seg_edu": seg_edu,
        "p_test": p,
        "p_late": p_late,
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    R.reset()

    stage("ingest: reading the six-month account panel")
    raw = F.load_raw()
    X_early, X_late, y = F.build_all()
    print(f"        {len(raw):,} accounts | default rate {y.mean():.4f}")

    # Segments for the responsible-AI check.  Bands, not raw values, because a
    # per-year breakdown is too thin to read.
    age_band = pd.cut(raw["AGE"], [20, 30, 40, 50, 60, 100],
                      labels=["21-30", "31-40", "41-50", "51-60", "61+"], right=True)
    edu_band = raw["EDUCATION"].map({1: "graduate", 2: "university", 3: "high school"}
                                    ).fillna("other")

    stage("features: building the identical feature set from both windows")
    print(f"        {len(F.FEATURES)} features | early = Apr-Jun, late = Jul-Sep")

    Xtr, Xte, ytr, yte = train_test_split(
        X_early, y, test_size=0.25, stratify=y, random_state=SEED)

    stage("reference: best single-feature rule, which sets the discrimination floor")
    single = []
    for col in X_early.columns:
        v = Xte[col].to_numpy(dtype=float)
        if np.all(np.isnan(v)):
            continue
        v = np.nan_to_num(v, nan=float(np.nanmedian(v)))
        a = M.auc(yte, v)
        single.append((col, max(a, 1 - a)))   # a feature that ranks backwards still ranks
    single.sort(key=lambda t: -t[1])
    best_single_feature, best_single_auc = single[0]
    floor = best_single_auc + 0.02
    print(f"        best single feature '{best_single_feature}' auc {best_single_auc:.4f} "
          f"-> discrimination floor {floor:.4f}")

    stage("train + evaluate: champion and challenger")
    evals, rows = {}, []
    for name, model in build_candidates().items():
        ev = evaluate(model, Xtr, ytr, Xte, yte, X_late, y, age_band, edu_band)
        evals[name] = (model, ev)
        m = ev["metrics"]
        print(f"        {name:<24} auc {m['auc']:.4f} | ece {m['ece']:.4f} "
              f"| late-window auc {m['auc_late_window']:.4f}")
        rows.append({"model": name, **{k: round(v, 4) for k, v in m.items()}})
    pd.DataFrame(rows).to_csv(RESULTS / "model_comparison.csv", index=False)

    stage("deployment gate: promoting only what clears every gate")
    entries = {}
    for name, (_, ev) in evals.items():
        entry = R.record(name, ev["metrics"],
                         {"seed": SEED, "cutoff_rate": CUTOFF_RATE,
                          "discrimination_floor": round(floor, 4),
                          "floor_source": f"best single feature ({best_single_feature}) + 0.02"},
                         overrides={"discrimination": floor})
        entries[name] = entry
        verdict = "PROMOTED" if entry["promoted"] else f"BLOCKED by {entry['blocked_by']}"
        print(f"        v{entry['version']} {name:<24} {verdict}")
        print(f"            {entry['challenger']['reason']}")

    champion_name = R.load()["champion"]["model"] if R.load()["champion"] else None
    if champion_name is None:
        print("        no candidate cleared the gates; nothing promoted")
        return 1
    champion, champ_eval = evals[champion_name]

    stage("drift: early window as baseline, late window as current")
    drift = M.feature_drift(X_early, X_late)
    drift.to_csv(RESULTS / "feature_drift.csv", index=False)
    score_psi = M.psi(champ_eval["p_test"], champ_eval["p_late"])
    moved = drift[drift["band"] != "stable"]
    print(f"        score PSI {score_psi:.4f} | {len(moved)} of {len(drift)} features "
          f"past the investigate threshold")

    stage("responsible AI: segment stability at the operating cutoff")
    champ_eval["seg_age"].to_csv(RESULTS / "segment_age.csv", index=False)
    champ_eval["seg_edu"].to_csv(RESULTS / "segment_education.csv", index=False)
    print(f"        widest false-decline gap "
          f"{champ_eval['metrics']['max_false_decline_gap']:.4f}")

    stage("serving: scoring in batches and writing the log")
    # The serving monitor's reference is the SERVING population, not the training
    # window.  Training-vs-serving drift is a slower, separate question, and the
    # drift stage above already answers it; mixing the two makes every batch look
    # broken on a portfolio that has genuinely moved.
    clean = S.score_batches(champion, X_late, n_batches=12, seed=SEED, reference=X_late)
    faulted = [S.inject_fault(champion, X_late, f, seed=SEED, reference=X_late)
               for f in S.FAULTS]
    S.write_log(clean + faulted)
    baseline = S.baseline_from(clean)

    stage("monitor: replaying the log through the alert rules")
    detail, caught = [], {"output_only": 0, "full": 0}
    for rec in faulted:
        out_only = S.monitor(rec, baseline, include_input_psi=False)
        full = S.monitor(rec, baseline, include_input_psi=True)
        caught["output_only"] += len(out_only) > 0
        caught["full"] += len(full) > 0
        detail.append({"fault": rec["fault"],
                       "detected_output_only": bool(out_only),
                       "detected_full": bool(full),
                       "rules": [a.rule for a in full]})
        mark = "detected" if full else "MISSED"
        extra = "" if out_only else "   <- invisible to output-only monitoring"
        print(f"        {rec['fault']:<14} {mark:<9} {[a.rule for a in full]}{extra}")

    fp_out = sum(len(S.monitor(r, baseline, include_input_psi=False)) > 0 for r in clean)
    fp_full = sum(len(S.monitor(r, baseline, include_input_psi=True)) > 0 for r in clean)
    n = len(faulted)
    print(f"        output-only rules: {caught['output_only']}/{n} faults, "
          f"{fp_out}/{len(clean)} clean batches alerting")
    print(f"        with input PSI:    {caught['full']}/{n} faults, "
          f"{fp_full}/{len(clean)} clean batches alerting")

    summary = {
        "accounts": int(len(raw)),
        "default_rate": round(float(y.mean()), 4),
        "n_features": len(F.FEATURES),
        "champion": champion_name,
        "champion_auc": round(champ_eval["metrics"]["auc"], 4),
        "champion_ece": round(champ_eval["metrics"]["ece"], 4),
        "auc_early_window": round(champ_eval["metrics"]["auc"], 4),
        "auc_late_window": round(champ_eval["metrics"]["auc_late_window"], 4),
        "score_psi_early_vs_late": round(float(score_psi), 4),
        "features_past_investigate": int(len(moved)),
        "worst_feature": drift.iloc[0]["feature"],
        "worst_feature_psi": float(drift.iloc[0]["psi"]),
        "best_single_feature": best_single_feature,
        "best_single_feature_auc": round(float(best_single_auc), 4),
        "discrimination_floor": round(float(floor), 4),
        "max_false_decline_gap": round(champ_eval["metrics"]["max_false_decline_gap"], 4),
        "faults_injected": len(faulted),
        "faults_detected_output_only": caught["output_only"],
        "faults_detected_with_input_psi": caught["full"],
        "clean_batches": len(clean),
        "clean_batches_alerting_output_only": fp_out,
        "clean_batches_alerting_full": fp_full,
        "fault_detail": detail,
        "blocked": {n: e["blocked_by"] for n, e in entries.items() if not e["promoted"]},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    stage("docs: regenerating the results page from the artifacts")
    import build_docs
    build_docs.main()

    stage("done: results/summary.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
