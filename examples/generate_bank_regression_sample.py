"""Generate a ~1000-row / 15-feature synthetic dataset for a BANK regression task.

Useful as a quick smoke-test / demo input for the AutoML pipeline with
`--model-type regression`.

Design (mimics a retail-bank "monthly profit per customer" prediction):
  - 12 informative features correlated with the target (income, credit score,
    balances, loan principal, product mix, region, ...).
  - 2 useless features (uniform random noise + a constant-ish junk column) to
    exercise the pipeline's feature-pruning path.
  - 1 feature that is ~95% null (`overdraft_amount`): only the rare customers
    who overdraw carry a value, everything else is NaN — exercises Stage 1b
    null processing.

Output: examples/bank_regression_sample.csv  (seedable, reproducible)

Usage:
  python examples/generate_bank_regression_sample.py            # writes CSV
  python examples/generate_bank_regression_sample.py --rows 1000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_DEFAULT = 42
N_ROWS_DEFAULT = 1000
OUTPUT_DEFAULT = Path(__file__).resolve().parent / "bank_regression_sample.csv"


def make_bank_regression_sample(n_rows: int = N_ROWS_DEFAULT,
                                seed: int = RNG_DEFAULT) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n = n_rows

    # ── informative features (each correlated with the target) ──────────────
    age = rng.normal(45, 12, n).clip(18, 90)
    monthly_income = rng.lognormal(mean=8.3, sigma=0.5, size=n)          # ~$4k median
    credit_score = rng.normal(680, 90, n).clip(300, 850)
    total_balance = rng.lognormal(mean=9.2, sigma=1.1, size=n)           # ~$9.9k median
    num_accounts = rng.integers(1, 11, size=n)
    num_transactions = rng.poisson(lam=22, size=n).clip(0, 200)
    loan_principal = rng.lognormal(mean=10.5, sigma=1.3, size=n) * rng.random(n)
    account_age_years = rng.integers(0, 31, size=n).astype(float)
    credit_history_length = rng.integers(0, 21, size=n).astype(float)

    region = rng.choice(["North", "South", "Central", "East", "West"], size=n, p=[0.25, 0.25, 0.2, 0.15, 0.15])
    product_type = rng.choice(
        ["Savings", "Checking", "CreditCard", "Mortgage", "PersonalLoan"],
        size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1],
    )
    occupation = rng.choice(
        ["Employed", "SelfEmployed", "Retired", "Unemployed", "Student"],
        size=n, p=[0.5, 0.15, 0.15, 0.1, 0.1],
    )

    # ── useless features (no relation to target) ────────────────────────────
    random_noise = rng.uniform(0.0, 1.0, size=n)
    useless_feature = rng.choice([0, 1, 2], size=n, p=[0.9, 0.07, 0.03])   # near-constant

    # ── ~95% null feature (rare event) ──────────────────────────────────────
    overdraft_amount = np.full(n, np.nan)
    n_overdraw = max(1, int(round(n * 0.05)))
    overdraft_idx = rng.choice(n, size=n_overdraw, replace=False)
    overdraft_amount[overdraft_idx] = rng.lognormal(mean=5.0, sigma=0.8, size=n_overdraw)

    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age.round(1),
        "monthly_income": monthly_income.round(2),
        "credit_score": credit_score.round(0),
        "total_balance": total_balance.round(2),
        "num_accounts": num_accounts,
        "num_transactions": num_transactions,
        "loan_principal": loan_principal.round(2),
        "account_age_years": account_age_years,
        "region": region,
        "product_type": product_type,
        "occupation": occupation,
        "credit_history_length": credit_history_length,
        "random_noise": random_noise.round(6),
        "useless_feature": useless_feature,
        "overdraft_amount": overdraft_amount.round(2),
    })

    # ── target: monthly_profit (regression), correlated with informative feats ──
    # Linear model on standardized drivers so scale stays ~$hundreds..~$20k.
    region_effect = df["region"].map({"North": 300, "South": 200, "Central": 100,
                                       "East": 0, "West": -150})
    product_effect = df["product_type"].map({"Mortgage": 900, "PersonalLoan": 600,
                                             "CreditCard": 300, "Checking": 50,
                                             "Savings": 0})
    occ_effect = df["occupation"].map({"Employed": 250, "SelfEmployed": 300,
                                       "Retired": 100, "Student": -50,
                                       "Unemployed": -200})
    # Each numeric driver is z-scored then weighted so every informative
    # feature contributes visibly and the target stays on a sane $-per-month scale.
    def _z(x):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + 1e-9)

    monthly_profit = (
        350 * _z(monthly_income)
        + 250 * _z(credit_score)
        + 260 * _z(np.log1p(total_balance))
        + 180 * _z(num_transactions)
        + 160 * _z(num_accounts)
        + 320 * _z(loan_principal)
        + 220 * _z(account_age_years)
        + 200 * _z(credit_history_length)
        + 130 * _z(age)
        + region_effect.fillna(0)
        + product_effect.fillna(0)
        + occ_effect.fillna(0)
        + rng.normal(0, 180, size=n)                         # irreducible noise
        + 0.5 * np.nan_to_num(overdraft_amount)              # rare negative-ish signal
    )
    df["monthly_profit"] = monthly_profit.clip(lower=0).round(2)

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=N_ROWS_DEFAULT,
                    help=f"Number of rows (default {N_ROWS_DEFAULT})")
    ap.add_argument("--seed", type=int, default=RNG_DEFAULT,
                    help=f"Random seed for reproducibility (default {RNG_DEFAULT})")
    ap.add_argument("--out", type=Path, default=OUTPUT_DEFAULT,
                    help=f"Output CSV path (default {OUTPUT_DEFAULT})")
    args = ap.parse_args()

    df = make_bank_regression_sample(n_rows=args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {args.out}")
    print(f"Null ratio (overdraft_amount): {df['overdraft_amount'].isna().mean():.1%}")
    print("Target (monthly_profit): "
          f"mean={df['monthly_profit'].mean():.0f} "
          f"min={df['monthly_profit'].min():.0f} "
          f"max={df['monthly_profit'].max():.0f}")
    print("Top |corr| with target:")
    corr = df.select_dtypes(include=[np.number]).corr()["monthly_profit"].drop("monthly_profit").abs().sort_values(ascending=False)
    for col, v in corr.head(6).items():
        print(f"  {col:<22} {v:.3f}")


if __name__ == "__main__":
    main()