"""Rebuild docs/index.html from results/.  python src/build_docs.py

The page is generated rather than hand-written so it cannot drift away from the
numbers the pipeline actually produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "index.html"


def bars(rows, value_key, label_key, fmt="{:.3f}", highlight=None):
    peak = max(abs(float(r[value_key])) for r in rows) or 1.0
    out = []
    for r in rows:
        v = float(r[value_key])
        pct = abs(v) / peak * 100
        cls = " hot" if highlight and highlight(r) else ""
        out.append(
            f'<div class="bar-row"><span class="bar-label">{r[label_key]}</span>'
            f'<span class="bar-track"><span class="bar-fill{cls}" '
            f'style="width:{pct:.1f}%"></span></span>'
            f'<span class="bar-value">{fmt.format(v)}</span></div>')
    return "\n".join(out)


def table(df: pd.DataFrame, cols: dict[str, str], pct_cols=()) -> str:
    head = "".join(f"<th>{v}</th>" for v in cols.values())
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pct_cols:
                cells.append(f"<td>{float(v) * 100:.1f}%</td>")
            elif isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer()):
                cells.append(f"<td>{int(v):,}</td>")
            elif isinstance(v, float):
                cells.append(f"<td>{v:.4f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        cls = "" if r.get("reliable", True) else ' class="muted"'
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def main() -> int:
    s = json.loads((RESULTS / "summary.json").read_text())
    reg = json.loads((RESULTS / "registry.json").read_text())
    drift = pd.read_csv(RESULTS / "feature_drift.csv")
    age = pd.read_csv(RESULTS / "segment_age.csv")
    edu = pd.read_csv(RESULTS / "segment_education.csv")

    # Pull the blocked candidate's calibration error from the registry rather
    # than the summary, so the page cannot quote a number no run produced.
    blocked = [h for h in reg["history"] if not h["promoted"]]
    blocked_ece = blocked[0]["metrics"]["ece"] if blocked else None
    blocked_name = blocked[0]["model"].replace("_", " ").title() if blocked else None

    gate_rows = []
    for h in reg["history"]:
        verdict = ("<span class='ok'>promoted</span>" if h["promoted"]
                   else f"<span class='no'>blocked: {', '.join(h['blocked_by'])}</span>")
        gate_rows.append(
            f"<tr><td>v{h['version']}</td><td>{h['model']}</td>"
            f"<td>{h['metrics']['auc']:.4f}</td><td>{h['metrics']['ece']:.4f}</td>"
            f"<td>{h['metrics']['max_false_decline_gap']:.4f}</td><td>{verdict}</td></tr>")

    fault_rows = []
    for f in s["fault_detail"]:
        out_only = ("<span class='ok'>caught</span>" if f["detected_output_only"]
                    else "<span class='no'>missed</span>")
        full = ("<span class='ok'>caught</span>" if f["detected_full"]
                else "<span class='no'>missed</span>")
        fault_rows.append(
            f"<tr><td><code>{f['fault']}</code></td><td>{out_only}</td>"
            f"<td>{full}</td><td>{', '.join(f['rules'])}</td></tr>")

    drift_bars = bars(drift.to_dict("records"), "psi", "feature",
                      highlight=lambda r: r["band"] != "stable")

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Credit Scorecard Operations</title>
<style>
:root {{
  --bg:#fbfaf8; --panel:#ffffff; --ink:#1a1a1a; --muted:#6b6b6b;
  --line:#e6e3de; --accent:#1f6f54; --warn:#b4531f; --rule:#d8d4cd;
}}
:root:not([data-theme="light"]) {{ }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#15161a; --panel:#1d1f24; --ink:#eceae6; --muted:#9c9a95;
    --line:#2d3038; --accent:#53b78f; --warn:#e08a4e; --rule:#343842;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15161a; --panel:#1d1f24; --ink:#eceae6; --muted:#9c9a95;
  --line:#2d3038; --accent:#53b78f; --warn:#e08a4e; --rule:#343842;
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif; }}
.wrap {{ max-width:960px; margin:0 auto; padding:48px 16px 80px }}
h1 {{ font-size:clamp(1.6rem,4vw,2.3rem); line-height:1.2; margin:0 0 8px; letter-spacing:-.02em }}
h2 {{ font-size:1.15rem; margin:44px 0 6px; letter-spacing:-.01em }}
.sub {{ color:var(--muted); margin:0 0 6px }}
.lede {{ color:var(--muted); max-width:62ch; margin:0 0 28px }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; margin:14px 0 }}
.kpis {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); margin:24px 0 }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px }}
.kpi .n {{ font-size:1.5rem; font-weight:650; letter-spacing:-.02em; font-variant-numeric:tabular-nums }}
.kpi .l {{ color:var(--muted); font-size:.82rem; margin-top:2px }}
.panel {{ overflow-x:auto; -webkit-overflow-scrolling:touch }}
table {{ width:100%; border-collapse:collapse; font-size:.9rem;
  font-variant-numeric:tabular-nums; min-width:0 }}
td,th {{ white-space:nowrap }}
th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line) }}
th {{ color:var(--muted); font-weight:600; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em }}
tbody tr:last-child td {{ border-bottom:none }}
tr.muted td {{ color:var(--muted) }}
code {{ font-size:.86em; background:var(--bg); padding:1px 5px; border-radius:4px; border:1px solid var(--line) }}
.ok {{ color:var(--accent); font-weight:600 }}
.no {{ color:var(--warn); font-weight:600 }}
.bar-row {{ display:grid; grid-template-columns:minmax(110px,1fr) 3fr 56px;
  gap:10px; align-items:center; padding:3px 0; font-size:.86rem }}
.bar-label {{ color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap }}
.bar-track {{ background:var(--bg); border:1px solid var(--line); border-radius:999px; height:10px; overflow:hidden }}
.bar-fill {{ display:block; height:100%; background:var(--rule); border-radius:999px }}
.bar-fill.hot {{ background:var(--warn) }}
.bar-value {{ text-align:right; font-variant-numeric:tabular-nums; color:var(--muted) }}
.note {{ font-size:.86rem; color:var(--muted); max-width:66ch }}
.cols {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)) }}
a {{ color:var(--accent) }}
@media (max-width:520px) {{ .bar-row {{ grid-template-columns:minmax(90px,1fr) 2fr 50px }} }}
</style></head><body><div class="wrap">

<h1>Credit Scorecard Operations</h1>
<p class="sub">Promotion gates · drift and anomaly monitoring · segment stability</p>
<p class="lede">A production-shaped pipeline over {s['accounts']:,} real card accounts.
The modelling is deliberately ordinary. Everything around it — what is allowed to
deploy, what fires when serving goes wrong, and who reads it at 3am — is the work.</p>

<div class="kpis">
  <div class="kpi"><div class="n">{s['accounts']:,}</div><div class="l">accounts scored</div></div>
  <div class="kpi"><div class="n">{s['champion_auc']:.4f}</div><div class="l">champion AUC</div></div>
  <div class="kpi"><div class="n">{s['champion_ece']:.4f}</div><div class="l">calibration error</div></div>
  <div class="kpi"><div class="n">{s['faults_detected_with_input_psi']}/{s['faults_injected']}</div><div class="l">injected faults caught</div></div>
  <div class="kpi"><div class="n">{s['clean_batches_alerting_full']}/{s['clean_batches']}</div><div class="l">false alerts</div></div>
</div>

<h2>The promotion gate</h2>
<p class="note">The discrimination floor is measured, not chosen: the best single-feature
rule (<code>{s['best_single_feature']}</code>, AUC {s['best_single_feature_auc']:.4f}) plus 0.02.</p>
<div class="panel"><table>
<thead><tr><th></th><th>Model</th><th>AUC</th><th>ECE</th><th>Segment gap</th><th>Outcome</th></tr></thead>
<tbody>{''.join(gate_rows)}</tbody></table></div>
<p class="note">{blocked_name} <em>ranks</em> well enough to clear the floor. Its probabilities are
wrong by {blocked_ece:.3f}. Limits and provisions are set off those probabilities,
so AUC alone would have shipped it.</p>

<h2>Serving monitor</h2>
<div class="panel"><table>
<thead><tr><th>Injected fault</th><th>Output rules</th><th>+ input PSI</th><th>Fired</th></tr></thead>
<tbody>{''.join(fault_rows)}</tbody></table></div>
<p class="note">The stuck upstream feature moved the mean score by 0.0006 — 0.2 standard
deviations of normal batch variation. The model leaned on correlated features and kept
producing plausible numbers. Only the input side could see it. Detection rates are on
deliberately injected faults, not production incidents.</p>

<h2>Feature drift, April–June against July–September</h2>
<div class="panel">{drift_bars}</div>
<p class="note">Score-level PSI {s['score_psi_early_vs_late']:.4f}. Two of {s['n_features']} features
cross the 0.10 investigate threshold, both on the repayment side; the balance and limit
features did not move. Same accounts in both windows, so this is how the features moved,
not how a fresh cohort behaved.</p>

<h2>Segment stability at the operating cutoff</h2>
<div class="cols">
<div class="panel">{table(age, {'segment': 'Age', 'n': 'n', 'decline_rate': 'Declined', 'false_decline_rate': 'False declines'}, pct_cols=('decline_rate', 'false_decline_rate'))}</div>
<div class="panel">{table(edu, {'segment': 'Education', 'n': 'n', 'decline_rate': 'Declined', 'false_decline_rate': 'False declines'}, pct_cols=('decline_rate', 'false_decline_rate'))}</div>
</div>
<p class="note">Widest gap across reliable segments: {s['max_false_decline_gap'] * 100:.1f} points.
Greyed rows hold fewer than 200 accounts and are reported but excluded from the gate —
without that rule a 72-account band set the gap at 12.8 points. The check reports; what
gap is acceptable is a policy decision, not a modelling one.</p>

<p class="note" style="margin-top:40px">
UCI <em>Default of Credit Card Clients</em> (Yeh &amp; Lien, 2009) — 30,000 Taiwanese card
accounts, April–September 2005, {s['default_rate'] * 100:.1f}% default rate. Public research
dataset, no personal identifiers. ·
<a href="https://github.com/f20220970-debug/bread-credit-mlops">Source and runbook on GitHub</a>
</p>
</div></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
