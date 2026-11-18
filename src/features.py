"""Feature construction for the two billing windows.

The source panel is UCI "Default of Credit Card Clients": 30,000 Taiwanese card
accounts observed over six consecutive billing months (April-September 2005),
with the label `default payment next month` recorded after the September bill.

Each account appears once, with six months of columns.  That lets us build the
SAME feature set twice from two different parts of the account's history:

    EARLY window  -> April, May, June     (PAY_6/5/4, BILL_AMT6/5/4, PAY_AMT6/5/4)
    LATE  window  -> July, August, Sept   (PAY_4/3/2, BILL_AMT3/2/1, PAY_AMT3/2/1)

This is what makes the drift in this project real rather than injected: the two
windows are genuinely different months of a real portfolio that was moving
through the 2005-06 Taiwan card-debt crisis.  It is NOT an out-of-time split in
the usual sense -- the accounts are the same in both windows and the label is
the same.  So the late window tells us how the FEATURES moved, not how a fresh
cohort behaved.  Every claim downstream is worded to respect that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "uci_default.xls"
LABEL = "default payment next month"

# (repayment-status cols, bill cols, payment cols) oldest -> newest in each window
WINDOWS = {
    "early": (["PAY_6", "PAY_5", "PAY_4"],
              ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4"],
              ["PAY_AMT6", "PAY_AMT5", "PAY_AMT4"]),
    "late": (["PAY_4", "PAY_3", "PAY_2"],
             ["BILL_AMT3", "BILL_AMT2", "BILL_AMT1"],
             ["PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]),
}

FEATURES = [
    "limit_bal", "age",
    "pay_status_latest", "pay_status_max", "pay_status_mean",
    "months_delinquent",
    "bill_latest", "bill_mean", "bill_trend",
    "paid_latest", "paid_mean",
    "utilisation", "payment_ratio", "balance_growth",
]


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Read the panel.  Row 0 of the sheet is a merged banner; real header is row 1."""
    df = pd.read_excel(path, header=1)
    return df.drop(columns=["ID"])


def build_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """Build the identical feature set from one three-month window."""
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; expected one of {sorted(WINDOWS)}")
    pay_cols, bill_cols, amt_cols = WINDOWS[window]

    pay = df[pay_cols].to_numpy(dtype=float)
    bill = df[bill_cols].to_numpy(dtype=float)
    amt = df[amt_cols].to_numpy(dtype=float)
    limit = df["LIMIT_BAL"].to_numpy(dtype=float)

    out = pd.DataFrame(index=df.index)
    out["limit_bal"] = limit
    out["age"] = df["AGE"].to_numpy(dtype=float)

    # Repayment status: -2/-1/0 = paid or revolving, 1..9 = months past due.
    out["pay_status_latest"] = pay[:, -1]
    out["pay_status_max"] = pay.max(axis=1)
    out["pay_status_mean"] = pay.mean(axis=1)
    out["months_delinquent"] = (pay > 0).sum(axis=1)

    out["bill_latest"] = bill[:, -1]
    out["bill_mean"] = bill.mean(axis=1)
    out["bill_trend"] = bill[:, -1] - bill[:, 0]

    out["paid_latest"] = amt[:, -1]
    out["paid_mean"] = amt.mean(axis=1)

    # Ratios, guarded: credit limits are always positive here, but bills can be
    # zero or negative (credit balances), so the payment ratio needs a floor.
    out["utilisation"] = bill[:, -1] / np.where(limit > 0, limit, np.nan)
    denom = np.where(bill[:, -2] > 0, bill[:, -2], np.nan)
    out["payment_ratio"] = np.clip(amt[:, -1] / denom, 0, 5)
    out["balance_growth"] = (bill[:, -1] - bill[:, 0]) / np.where(limit > 0, limit, np.nan)

    out = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def build_all(path: Path = RAW):
    """Return (early_X, late_X, y) with a shared index."""
    df = load_raw(path)
    y = df[LABEL].astype(int)
    return build_window(df, "early"), build_window(df, "late"), y
