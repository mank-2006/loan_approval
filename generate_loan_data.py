"""Create fictional, affordability-aware example applications for the demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from loan_approval_model import estimate_monthly_payment


OUTPUT_PATH = Path(__file__).with_name("loan_data_sample.csv")
ROW_COUNT = 4_000
RANDOM_SEED = 20261118


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def build_examples() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    monthly_income = np.exp(rng.uniform(np.log(10_000), np.log(1_000_000), ROW_COUNT))
    coapplicant_income = np.where(
        rng.random(ROW_COUNT) < 0.62,
        0,
        np.exp(rng.uniform(np.log(8_000), np.log(300_000), ROW_COUNT)),
    )
    loan_amount = np.exp(rng.uniform(np.log(10_000), np.log(3_000_000), ROW_COUNT))
    loan_term = rng.choice([12, 24, 36, 60, 84, 120, 180, 240, 300], ROW_COUNT)
    cibil_score = rng.integers(300, 901, ROW_COUNT)
    employment = rng.choice(
        ["Employed", "Self-Employed", "Unemployed"],
        ROW_COUNT,
        p=[0.62, 0.28, 0.10],
    )
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], ROW_COUNT)

    monthly_payment = np.array(
        [estimate_monthly_payment(amount, term) for amount, term in zip(loan_amount, loan_term)]
    )
    household_income = np.maximum(monthly_income + coapplicant_income, 1)
    repayment_ratio = monthly_payment / household_income

    # Illustrative labels: credit score matters, while very affordable payments
    # improve the odds. The rates and thresholds are teaching assumptions, not
    # learned from a bank or real applicant decisions.
    employment_adjustment = np.select(
        [employment == "Employed", employment == "Self-Employed"],
        [0.25, 0.05],
        default=-0.65,
    )
    approval_log_odds = (
        0.25
        + (cibil_score - 650) / 100
        + 1.2 * (0.25 - repayment_ratio)
        - 5 * np.maximum(repayment_ratio - 0.25, 0)
        + employment_adjustment
    )
    approval_probability = sigmoid(approval_log_odds)
    status = np.where(rng.random(ROW_COUNT) < approval_probability, "Y", "N")

    examples = pd.DataFrame(
        {
            "Application_ID": [f"DEMO{i:05d}" for i in range(1, ROW_COUNT + 1)],
            "ApplicantIncome": monthly_income.round().astype(int),
            "CoapplicantIncome": coapplicant_income.round().astype(int),
            "LoanAmount": loan_amount.round().astype(int),
            "Loan_Amount_Term": loan_term.astype(int),
            "CIBIL_Score": cibil_score,
            "Employment_Status": employment,
            "Property_Area": property_area,
            "Loan_Status": status,
        }
    )

    # Keep the user's example visible in the sample file and label it approved.
    example_row = pd.DataFrame(
        [
            {
                "Application_ID": "DEMO-EXAMPLE",
                "ApplicantIncome": 1_000_000,
                "CoapplicantIncome": 0,
                "LoanAmount": 10_000,
                "Loan_Amount_Term": 12,
                "CIBIL_Score": 670,
                "Employment_Status": "Employed",
                "Property_Area": "Urban",
                "Loan_Status": "Y",
            }
        ]
    )
    return pd.concat([example_row, examples], ignore_index=True)


if __name__ == "__main__":
    data = build_examples()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {len(data):,} fictional examples at {OUTPUT_PATH}")
    print(data["Loan_Status"].value_counts().rename(index={"Y": "Approved", "N": "Rejected"}))
