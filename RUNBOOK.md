# Runbook: credit scorecard, batch scoring

For whoever is on call when an alert fires. Each entry is written to be worked
through without retraining anything and without reading the modelling code.

The alert rules live in `src/serving.py`. The thresholds they compare against
are rebuilt from the healthy batches of each run (`baseline_from`), so they move
with the portfolio rather than being hard-coded.

---

## `input_drift` — a feature's distribution has moved

**Alert text:** `input PSI 5.998 on 'pay_status_latest' against the serving baseline`

**What it means.** The named feature arrived with a different distribution from
the recent serving population. PSI above 0.25 is the act threshold; 0.10–0.25 is
worth investigating but is not on its own a reason to stop scoring.

**Most likely causes, cheapest to check first.**

1. **An upstream job failed and the feature is stuck on a constant.** Check
   `score_std` in the same log line — a stuck input often leaves the score
   spread intact, which is why this rule exists separately from the score rules.
   Confirm by grouping the batch's raw feature values: a single value holding
   most of the mass is conclusive.
2. **A join dropped rows or widened.** Compare `n` against the neighbouring
   batches in `results/serving_log.jsonl`.
3. **A genuine portfolio shift.** If the feature is a repayment-behaviour one
   and the move is gradual across several batches rather than a step change in
   one, this is the portfolio moving, not a bug. Escalate to the model owner;
   do not treat it as an incident.

**First action.** Do not fail the batch on input drift alone unless PSI is above
1.0, which in practice means a constant or an empty column. Hold the batch,
confirm against the previous run, and page the owning data team with the feature
name and the PSI.

---

## `null_rate` — a column arrived empty or mostly empty

**Alert text:** `null rate 0.079 against a baseline of 0.008`

**What it means.** The share of missing values across the feature matrix jumped.
The model will still score — the imputer fills the gap — which is precisely the
danger: the output looks normal.

**Causes.** A schema change upstream, a renamed column, a failed join, a
partition that did not land.

**First action.** Stop the batch. A scored population with a silently imputed
feature produces decisions that cannot be explained afterwards. Identify the
column from the paired `input_drift` alert, which names it.

---

## `score_shift` — the score distribution moved

**Alert text:** `mean score 0.2752 is 13.7 sd from the baseline mean 0.2289`

**What it means.** The batch's mean score is far from the recent healthy mean,
measured in standard deviations of the batch-to-batch variation.

**Causes.** In order of how often they turn out to be the answer: a units or
scaling change in an input (the fault-injection suite's `score_shift` case is a
credit limit arriving in thousands); a model version swapped without the
reference being rebuilt; a real change in population mix; a genuine change in
risk.

**First action.** Check whether `input_drift` fired on the same batch. If it did,
the cause is upstream and the score shift is a symptom — work the input alert.
If the score moved and no input did, suspect the model artefact itself: confirm
the deployed version against `results/registry.json`.

---

## `latency` — scoring slowed down

**Alert text:** `20.1 ms/1k against a p95 of 3.3`

**What it means.** The batch took more than three times the recent p95 per
thousand rows.

**Causes.** Feature store or warehouse contention, a cold start, an unusually
large batch, resource pressure on the host.

**First action.** Check `n` first — a bigger batch is not an incident. If the
per-1k cost genuinely rose, this is an infrastructure alert, not a model one.

---

## `score_collapse` — the score spread has flattened

**What it means.** The spread of scores within the batch fell below half the
healthy spread. A model that returns nearly the same number for everyone is not
ranking anything.

**Causes.** A dominant feature stuck on a constant; a model artefact that failed
to load and is falling back to a prior; an all-null feature matrix.

**First action.** Stop the batch. This one is never benign.

---

## What this monitor does not cover

Stated plainly, because the gaps matter more than the coverage:

- **No label feedback.** Defaults are known months after scoring, so nothing
  here measures live accuracy. Drift and anomaly detection are leading
  indicators and nothing more.
- **Batches are units of work, not days.** The source panel carries no calendar
  column, so there is no time-of-day or day-of-week seasonality to model.
- **Detection rates in this repo are on deliberately injected faults** (see
  `src/serving.py`). They say the rules fire on the failure modes they were
  written for. They are not a claim about production.
- **The segment-stability check reports; it does not adjudicate.** The gap
  thresholds that would make a disparity acceptable are a policy decision.
