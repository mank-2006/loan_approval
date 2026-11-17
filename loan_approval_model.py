"""Train a loan approval classifier from a labelled CSV file.

Expected target values include Y/N, Yes/No, Approved/Rejected, or 1/0.
The model includes an estimated repayment-to-income feature when the required
income, loan amount, and term columns are present.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Common spellings are supported. If your CSV uses different headers, pass
# them with --features and --target when running this script.
FEATURE_ALIASES = {
    "ApplicantIncome": ["ApplicantIncome", "Applicant_Income", "applicant_income"],
    "LoanAmount": ["LoanAmount", "Loan_Amount", "loan_amount"],
    "CIBIL_Score": ["CIBIL_Score", "CIBILScore", "CIBIL", "Credit_Score", "credit_score"],
    "Employment_Status": [
        "Employment_Status",
        "EmploymentStatus",
        "Self_Employed",
        "SelfEmployed",
        "employment_status",
    ],
    "Property_Area": ["Property_Area", "PropertyArea", "property_area"],
}

OPTIONAL_FINANCIAL_FEATURES = [
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Monthly_Installment",
    "Existing_Debt",
    "Debt_To_Income_Ratio",
]

APPROVED_LABELS = {"y", "yes", "approved", "approve", "accepted", "accept", "1", "1.0", "true"}
REJECTED_LABELS = {"n", "no", "rejected", "reject", "declined", "decline", "0", "0.0", "false"}
DEMO_ANNUAL_INTEREST_RATE = 0.12
REPAYMENT_RATIO_FEATURE = "Estimated_Repayment_to_Income_Ratio"


def normalize_name(value: str) -> str:
    """Normalize column names for case- and punctuation-insensitive matching."""
    return "".join(char.lower() for char in value if char.isalnum())


def resolve_column(requested: str, columns: list[str]) -> str | None:
    by_normalized_name = {normalize_name(column): column for column in columns}
    return by_normalized_name.get(normalize_name(requested))


def estimate_monthly_payment(
    loan_amount: float,
    term_months: float,
    annual_interest_rate: float = DEMO_ANNUAL_INTEREST_RATE,
) -> float:
    """Estimate a fixed monthly payment for the classroom affordability feature."""
    try:
        principal = float(loan_amount)
        term = float(term_months)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(principal) or not math.isfinite(term):
        return 0.0
    principal = max(principal, 0.0)
    months = int(term)
    if principal == 0 or months <= 0:
        return 0.0

    monthly_rate = annual_interest_rate / 12
    return principal * monthly_rate / (1 - (1 + monthly_rate) ** -months)


def add_affordability_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add estimated payment as a share of combined monthly income when possible."""
    result = data.copy()
    columns = result.columns.tolist()
    income_column = resolve_column("ApplicantIncome", columns)
    coapplicant_column = resolve_column("CoapplicantIncome", columns)
    amount_column = resolve_column("LoanAmount", columns)
    term_column = resolve_column("Loan_Amount_Term", columns)

    if income_column is None or amount_column is None or term_column is None:
        return result

    applicant_income = pd.to_numeric(result[income_column], errors="coerce").fillna(0)
    coapplicant_income = (
        pd.to_numeric(result[coapplicant_column], errors="coerce").fillna(0)
        if coapplicant_column
        else 0
    )
    household_income = (applicant_income + coapplicant_income).clip(lower=1)
    payments = pd.Series(
        [
            estimate_monthly_payment(amount, term)
            for amount, term in zip(result[amount_column], result[term_column])
        ],
        index=result.index,
        dtype="float64",
    )
    result[REPAYMENT_RATIO_FEATURE] = (payments / household_income).clip(upper=10)
    return result


def resolve_features(
    columns: list[str], explicit_features: list[str] | None
) -> list[str]:
    if explicit_features:
        missing = [name for name in explicit_features if resolve_column(name, columns) is None]
        if missing:
            raise ValueError(
                f"Feature column(s) not found: {', '.join(missing)}. "
                f"CSV columns: {', '.join(columns)}"
            )
        return list(dict.fromkeys(resolve_column(name, columns) for name in explicit_features))

    selected: list[str] = []
    missing_concepts: list[str] = []
    for concept, aliases in FEATURE_ALIASES.items():
        match = next((resolve_column(alias, columns) for alias in aliases if resolve_column(alias, columns)), None)
        if match is None:
            missing_concepts.append(concept)
        else:
            selected.append(match)

    if missing_concepts:
        raise ValueError(
            "Could not find a column for these required project features: "
            f"{', '.join(missing_concepts)}. Use --features to specify your CSV's feature columns."
        )

    for optional_name in OPTIONAL_FINANCIAL_FEATURES:
        match = resolve_column(optional_name, columns)
        if match and match not in selected:
            selected.append(match)
    return selected


def encode_target(target: pd.Series) -> pd.Series:
    """Convert common approval labels to 1 (approved) and 0 (rejected)."""
    encoded = []
    unknown = set()
    for value in target:
        if pd.isna(value):
            encoded.append(pd.NA)
            continue
        label = str(value).strip().lower()
        if label in APPROVED_LABELS:
            encoded.append(1)
        elif label in REJECTED_LABELS:
            encoded.append(0)
        else:
            unknown.add(str(value))
            encoded.append(pd.NA)

    if unknown:
        raise ValueError(
            "Unrecognized target value(s): " + ", ".join(sorted(unknown))
            + ". Expected labels such as Approved/Rejected, Yes/No, Y/N, or 1/0."
        )
    return pd.Series(encoded, index=target.index, dtype="Int64")


def make_pipeline(data: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    numeric_columns = data[feature_columns].select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a binary loan approval classifier."
    )
    parser.add_argument("--data", required=True, help="Path to the labelled training CSV")
    parser.add_argument("--target", default="Loan_Status", help="Target column (default: Loan_Status)")
    parser.add_argument(
        "--features",
        help="Comma-separated feature columns; by default, the project columns are detected automatically",
    )
    parser.add_argument(
        "--model-out", default="loan_approval_model.joblib", help="Path for the saved model"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")

    data_path = Path(args.data)
    if not data_path.is_file():
        parser.error(f"Training CSV not found: {data_path}")

    try:
        data = pd.read_csv(data_path)
        target_column = resolve_column(args.target, data.columns.tolist())
        if target_column is None:
            raise ValueError(
                f"Target column '{args.target}' not found. CSV columns: {', '.join(data.columns)}"
            )

        explicit_features = args.features.split(",") if args.features else None
        if explicit_features:
            explicit_features = [name.strip() for name in explicit_features if name.strip()]
        feature_columns = resolve_features(data.columns.tolist(), explicit_features)

        # Exclude rows without a known training outcome.
        y_all = encode_target(data[target_column])
        valid_rows = y_all.notna()
        X = data.loc[valid_rows, feature_columns].copy()
        y = y_all.loc[valid_rows].astype(int)

        if y.nunique() != 2:
            raise ValueError("Training data must contain both approved and rejected applications.")
        if y.value_counts().min() < 2:
            raise ValueError("Each target class needs at least two rows for a stratified split.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=y,
        )

        X_train_model = add_affordability_features(X_train)
        X_test_model = add_affordability_features(X_test)
        model_feature_columns = X_train_model.columns.tolist()

        model = make_pipeline(X_train_model, model_feature_columns)
        model.fit(X_train_model, y_train)
        predictions = model.predict(X_test_model)
        probabilities = model.predict_proba(X_test_model)[:, 1]

        print(f"Rows used: {len(X)} (training: {len(X_train)}, test: {len(X_test)})")
        print("Features:", ", ".join(feature_columns))
        print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
        print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
        print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
        print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.3f}")
        print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.3f}")
        print("Confusion matrix (actual rows: rejected, approved; predicted columns: rejected, approved):")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))
        print("\nClassification report:")
        print(classification_report(y_test, predictions, labels=[0, 1], target_names=["Rejected", "Approved"], zero_division=0))

        # Keep the holdout metrics above, then train the saved model on all
        # labelled rows so no usable training examples are discarded.
        X_model = add_affordability_features(X)
        model_feature_columns = X_model.columns.tolist()
        final_model = make_pipeline(X_model, model_feature_columns)
        final_model.fit(X_model, y)
        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": final_model,
                "features": feature_columns,
                "model_features": model_feature_columns,
                "target": target_column,
                "class_names": {0: "Rejected", 1: "Approved"},
                "demo_interest_rate": DEMO_ANNUAL_INTEREST_RATE,
            },
            model_path,
        )
        print(f"\nSaved model to: {model_path.resolve()}")
        print("This model reflects patterns in the training data; it does not establish why past decisions were made.")
        return 0
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
