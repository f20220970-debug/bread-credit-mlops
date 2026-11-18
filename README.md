# Credit Scorecard Operations

**A production-shaped pipeline for a card portfolio: model registry with promotion gates, drift and anomaly monitoring on serving logs, segment-stability reporting, and a runbook for whoever is on call.**

Built for an AI Operations application. The modelling here is deliberately ordinary — the point is everything that surrounds a model once it has to run every day and be answered for.

📊 **[Results dashboard](https://f20220970-debug.github.io/bread-credit-mlops/)** · [Runbook](RUNBOOK.md) · [Pipeline](pipeline/run.py)

---

## The data

UCI *Default of Credit Card Clients* (Yeh & Lien, 2009): **30,000 real Taiwanese card accounts**, six consecutive billing months (April–September 2005), labelled with whether the account defaulted the following month. Default rate **22.1%**. Public research dataset, no personal identifiers.

The six-month panel is what makes this repo work. The same 14-feature set is built twice from two different parts of each account's history:

| Window | Months | Columns |
| --- | --- | --- |
| `early` | April–June | `PAY_6/5/4`, `BILL_AMT6/5/4`, `PAY_AMT6/5/4` |
| `late` | July–September | `PAY_4/3/2`, `BILL_AMT3/2/1`, `PAY_AMT3/2/1` |

**What that is and is not.** The drift measured between those windows is real: it is a real portfolio moving through the 2005 Taiwan card-debt crisis. It is **not** an out-of-time split. The accounts are the same in both windows and the label is the same, so the late window says how the *features* moved, not how a fresh cohort behaved. Every claim below is worded to respect that.

## Results

Everything here is written by `pipeline/run.py` into `results/`, and CI reruns it on every push.

### The gate decided three candidates differently

| Version | Model | AUC | ECE | Outcome |
| --- | --- | --- | --- | --- |
| v1 | Logistic regression | 0.6853 | 0.0134 | Promoted — no incumbent, becomes the baseline champion |
| v2 | Gradient boosting | **0.7135** | **0.0115** | Promoted — **+0.0282 AUC** over the champion, no regression |
| v3 | Naive Bayes | 0.6760 | **0.1249** | **Blocked on calibration** |

The third row is the one worth reading. Naive Bayes **ranks** acceptably — its AUC clears the discrimination floor — but its probabilities are wrong by **12.5 points**. A pipeline that promotes on AUC alone ships it. Credit limits and provisions are set off predicted probabilities, so a model whose 30% means 18% is not a smaller version of a good model; it is a different kind of broken.

**The discrimination floor is measured, not chosen.** It is the AUC of the best single-feature rule on the same test set (`months_delinquent`, AUC **0.6311**) plus 0.02. The gate asks whether a model beats the obvious thing it replaces, rather than whether it clears a number someone picked.

### The serving monitor missed a fault until it watched inputs as well as outputs

Four faults are injected on purpose to test the rules — a dropped column, a stuck upstream feature, a units error, and a slow feature store.

| Monitor | Faults caught | Clean batches alerting |
| --- | --- | --- |
| Output rules only (score, nulls, latency) | **3 / 4** | 0 / 12 |
| Plus per-feature input PSI | **4 / 4** | 0 / 12 |

The miss was the **stuck upstream feature**: `pay_status_latest` pinned to a constant. The mean score moved by **0.0006** — 0.2 standard deviations of normal batch-to-batch variation — and the spread moved by 0.006. The model simply leaned on correlated features and kept producing plausible numbers. Nothing on the output side could see it. Watching input distributions caught it at **PSI 6.0** on the stuck feature, and added no false alerts on the twelve healthy batches.

### Drift between the two billing windows

Score-level PSI is **0.0136** — stable. Two of fourteen features cross the 0.10 investigate threshold: `payment_ratio` (PSI **0.2357**) and `paid_latest` (PSI **0.1308**). Both are payment-behaviour features, and both moved while the balance and limit features did not. The repayment side of the portfolio changed; the exposure side did not.

A model trained on the April–June window scores **AUC 0.7507** on July–September, **above** its 0.7135 on its own window. That is the opposite of decay, and the reason is structural rather than encouraging: the label is default in October, so later months are simply closer to the event. It is recorded here because it is what the run produced.

### Segment stability at the operating cutoff

At a cutoff that declines the riskiest 20%, false-decline rates across reliable segments:

| Age band | n | False declines | | Education | n | False declines |
| --- | --- | --- | --- | --- | --- | --- |
| 31–40 | 2,655 | 11.1% | | Graduate | 2,649 | 10.3% |
| 21–30 | 2,752 | 14.6% | | University | 3,540 | 15.2% |
| 41–50 | 1,519 | 15.3% | | High school | 1,196 | 18.6% |
| 51–60 | 502 | 17.8% | | | | |

Widest gap across reliable segments: **8.3 points**, inside the 15-point gate.

Segments thinner than 200 accounts are reported but excluded from the gate. Without that rule the 61+ band (n=72, 23.9%) set the gap at **12.8 points** on 72 people — a number that swings on single cases. Letting a thin segment drive a deployment decision is its own failure mode.

The check reports; it does not adjudicate. What gap is acceptable is a policy decision, not a modelling one.

## Pipeline

```
ingest → features → reference → train + evaluate → promotion gate
       → drift → segment stability → batch scoring → monitor
```

Each stage prints its own status line and writes its own artefact, so a failure names the stage. `results/registry.json` holds every candidate ever evaluated, its metrics, every gate result and the reason it was or was not promoted.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/get_data.py      # ~5 MB
python pipeline/run.py
pytest                      # 32 tests
```

## Layout

| Path | Contents |
| --- | --- |
| `src/features.py` | The two-window feature build, with the honest framing of what the windows are |
| `src/monitoring.py` | PSI, KS, calibration error, segment stability |
| `src/registry.py` | Promotion gates, challenger rules, the registry file |
| `src/serving.py` | Batch scoring, serving logs, fault injection, alert rules |
| `pipeline/run.py` | The orchestration that chains all of it |
| `tests/` | 32 tests, run in CI |
| `RUNBOOK.md` | Per-alert triage: what it means, likely causes, first action |

## Tests

32 tests, all in CI, written against cases with a known answer rather than against a previous run's output: PSI is zero on identical samples and grows with the shift; calibration error is ~0 for a perfect forecaster and reads 0.2 on a 0.2 overstatement; **every gate is tested for blocking its own failure and only its own**; a missing metric fails closed rather than passing by default; AUC gain cannot buy a calibration or segment regression. CI additionally reruns the whole pipeline and fails if no champion is promoted or if the monitor misses an injected fault.

## What this does not claim

- **No label feedback loop.** Defaults are known months after scoring. Drift and anomaly detection are leading indicators, nothing more.
- **Detection rates are on injected faults**, not production incidents.
- **Batches are units of work, not days.** The panel has no calendar column.
- **One dataset, one portfolio, one market, 2005.** Thresholds here are defensible for this data and would have to be re-derived for another book.

## Source

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
