"""Streamlit website for the loan approval classroom demo."""

from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("loan_approval_model.joblib")

FEATURE_LABELS = {
    "ApplicantIncome": "Applicant monthly income",
    "CoapplicantIncome": "Co-applicant monthly income",
    "LoanAmount": "Requested loan amount",
    "Loan_Amount_Term": "Loan repayment term (months)",
    "CIBIL_Score": "CIBIL score (300–900)",
    "Employment_Status": "Applicant employment status",
    "Self_Employed": "Self-employed status",
    "Property_Area": "Property area",
    "Monthly_Installment": "Monthly installment",
    "Existing_Debt": "Existing debt",
    "Debt_To_Income_Ratio": "Debt-to-income ratio",
}

RUPEE_FEATURES = {
    "applicantincome",
    "coapplicantincome",
    "loanamount",
    "monthlyinstallment",
    "existingdebt",
}

FIELD_HELP = {
    "ApplicantIncome": "Enter the applicant's monthly income in Indian rupees.",
    "CoapplicantIncome": "Enter the co-applicant's monthly income in Indian rupees.",
    "LoanAmount": "Enter the total loan amount requested in Indian rupees.",
    "Loan_Amount_Term": "Enter the term in the units used by the training data, usually months.",
    "CIBIL_Score": "Enter the applicant's CIBIL score, from 300 to 900.",
    "Property_Area": "Select one of the area categories present in the training data.",
}

FIELD_CAPTIONS = {
    "ApplicantIncome": "Applicant's income per month. Example: ₹6,000 here means monthly income, not the requested loan amount.",
    "CoapplicantIncome": "Co-applicant's income per month in rupees. This is separate from the applicant's income.",
    "LoanAmount": "The total amount the applicant is requesting. Example: ₹1,80,000 here means loan amount, not income.",
    "Loan_Amount_Term": "How long the applicant requests to repay the loan. Use the units in your CSV, commonly months.",
    "CIBIL_Score": "Enter a CIBIL score between 300 and 900. This demo uses fictional scores; do not enter real applicant financial details.",
    "Employment_Status": "Choose the applicant's employment category.",
    "Self_Employed": "Choose the self-employment value used in your CSV.",
    "Property_Area": "Choose where the property is located.",
    "Monthly_Installment": "The expected monthly loan repayment in Indian rupees.",
    "Existing_Debt": "The applicant's existing debt amount in Indian rupees.",
    "Debt_To_Income_Ratio": "Debt compared with income. For example, 0.25 represents 25% if your CSV uses ratios this way.",
}


def normalized_name(column: str) -> str:
    return "".join(character.lower() for character in column if character.isalnum())


def pretty_name(column: str) -> str:
    readable = re.sub(r"(?<!^)(?=[A-Z])", " ", column.replace("_", " "))
    return FEATURE_LABELS.get(column, readable)


def to_python(value):
    return value.item() if hasattr(value, "item") else value


def section_for(column: str) -> str:
    name = normalized_name(column)
    if name in {"applicantincome", "coapplicantincome", "cibilscore", "employmentstatus", "selfemployed"}:
        return "Applicant profile"
    if name == "propertyarea":
        return "Property details"
    if name in {"loanamount", "loanamountterm", "monthlyinstallment"}:
        return "Loan request"
    return "Other financial details"


st.set_page_config(
    page_title="Loan approval demo",
    page_icon="🏦",
    layout="centered",
)

st.markdown(
    """
    <style>
      .stApp { background: #f4f7f5; }
      .block-container { max-width: 1020px; padding-top: 2.2rem; padding-bottom: 3rem; }
      [data-testid="stHeader"] { background: rgba(244,247,245,.8); }
      .hero {
        padding: 1.8rem 2rem; border-radius: 22px; color: #fff;
        background: linear-gradient(120deg, #123d2b 0%, #176b49 62%, #2c8c61 100%);
        box-shadow: 0 16px 38px rgba(18,61,43,.16); margin-bottom: 1.2rem;
      }
      .hero-top { display:flex; justify-content:space-between; align-items:center; gap:1rem; }
      .eyebrow { color:#b8e4c9; text-transform:uppercase; letter-spacing:.13em;
        font-size:.72rem; font-weight:750; margin:0 0 .55rem; }
      .badge { border:1px solid rgba(255,255,255,.35); border-radius:999px;
        padding:.38rem .7rem; color:#e7fff0; font-size:.7rem; font-weight:700;
        letter-spacing:.06em; white-space:nowrap; }
      .hero h1 { color:#fff; font-size:2.15rem; line-height:1.1; margin:0; letter-spacing:-.04em; }
      .hero p { color:#d9eee1; font-size:1rem; margin:.7rem 0 0; max-width:680px; }
      .steps { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.25rem; }
      .step { border-radius:999px; background:rgba(255,255,255,.12); padding:.42rem .72rem;
        color:#f2fff6; font-size:.78rem; }
      .section-title { color:#173e2d; font-weight:700; font-size:1.02rem; margin:.15rem 0 .1rem; }
      .result-card { border-radius:18px; padding:1.35rem 1.5rem; margin:.4rem 0 1rem; }
      .result-approved { background:#eaf7ef; border:1px solid #c5e7d0; }
      .result-rejected { background:#fff0ee; border:1px solid #f3d0ca; }
      .result-label { color:#64736a; font-size:.75rem; font-weight:750; letter-spacing:.1em; margin:0; }
      .result-title { color:#183e2d; font-size:1.8rem; font-weight:750; margin:.2rem 0 .1rem; }
      .result-copy { color:#586a60; margin:0; }
      div[data-testid="stForm"] { background:#fff; border:1px solid #e2e9e4;
        border-radius:18px; padding:1.25rem 1.35rem 1rem; }
      div[data-testid="stFormSubmitButton"] button {
        background:#176b49; color:white; border:0; border-radius:10px;
        min-height:3rem; font-weight:700; }
      div[data-testid="stFormSubmitButton"] button:hover { background:#10583a; color:white; }
      div[data-testid="stAlert"] { border-radius:12px; }
      .fine-print { color:#6b7970; font-size:.82rem; line-height:1.5; }
      [data-testid="stAppViewContainer"] { color:#17271e; }
      [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
        color:#183728 !important; font-weight:700 !important;
      }
      [data-testid="stCaptionContainer"] p {
        color:#344b3d !important; font-size:.86rem !important;
      }
      [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
        color:#253b2e;
      }
      [data-testid="stNumberInput"] input,
      [data-testid="stTextInput"] input,
      [data-baseweb="select"] input {
        color:#14251b !important; background:#ffffff !important;
        -webkit-text-fill-color:#14251b !important;
      }
      [data-testid="stNumberInput"] input,
      [data-testid="stTextInput"] input,
      [data-baseweb="select"] > div { border-color:#718579 !important; }
      [data-testid="stNumberInput"] div[role="button"] { color:#253b2e !important; }
      .hero p { color:#e7f4eb !important; }
      .hero .eyebrow { color:#c9efd7 !important; }
      .hero .badge { color:#ffffff !important; }
      .result-copy { color:#344b3d !important; }
      .fine-print { color:#43564a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-top">
        <div>
          <div class="eyebrow">AI for Engineers · Project demo</div>
          <h1>Loan approval preview</h1>
        </div>
        <div class="badge">INR · MONTHLY INCOME · BUILD 8 · CIBIL CHECK</div>
      </div>
      <p>Enter an application profile to see how the trained classifier responds. The form is grouped into simple sections, and every value can be changed before you submit.</p>
      <div class="steps">
        <span class="step">1 &nbsp; Add details</span>
        <span class="step">2 &nbsp; Get prediction</span>
        <span class="step">3 &nbsp; Review the estimate</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("The model file is missing. The website needs loan_approval_model.joblib beside app.py.")
    st.stop()

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["features"]
except Exception as exc:
    st.error(f"The model could not be opened: {exc}")
    st.stop()

preprocessor = model.named_steps["preprocessor"]
numeric_columns = list(preprocessor.transformers_[0][2])
categorical_columns = list(preprocessor.transformers_[1][2])

numeric_pipeline = preprocessor.named_transformers_["numeric"]
numeric_medians = dict(zip(numeric_columns, numeric_pipeline.named_steps["imputer"].statistics_))

categorical_pipeline = preprocessor.named_transformers_["categorical"]
encoder = categorical_pipeline.named_steps["onehot"]
categorical_choices = {
    column: [to_python(value) for value in categories]
    for column, categories in zip(categorical_columns, encoder.categories_)
}

st.markdown("### Application details")
st.info("All money amounts are in Indian rupees (₹). Applicant and co-applicant income are monthly. Enter a CIBIL score between 300 and 900.")

grouped_features = {}
for feature in feature_columns:
    grouped_features.setdefault(section_for(feature), []).append(feature)

section_notes = {
    "Applicant profile": "Monthly income, credit history, and employment information.",
    "Loan request": "The requested amount and any repayment details in the dataset.",
    "Property details": "Choose from the property-area categories learned by the model.",
    "Other financial details": "Additional financial fields available in the training data.",
}

submitted_values = {}
with st.form("loan_application_form"):
    for section in ["Applicant profile", "Loan request", "Property details", "Other financial details"]:
        fields = grouped_features.get(section, [])
        if not fields:
            continue

        st.markdown(f'<div class="section-title">{section}</div>', unsafe_allow_html=True)
        st.caption(section_notes[section])
        columns = st.columns(2, gap="large")

        for index, column in enumerate(fields):
            with columns[index % 2]:
                label = pretty_name(column)
                name = normalized_name(column)
                if name in RUPEE_FEATURES:
                    label += " (₹)"
                help_text = FIELD_HELP.get(column, "Use the same value format as the training data.")
                st.markdown(f"**{label}**")

                if column in categorical_columns:
                    choices = categorical_choices.get(column, [])
                    if choices:
                        submitted_values[column] = st.selectbox(
                            label,
                            options=choices,
                            format_func=str,
                            help=help_text,
                            label_visibility="collapsed",
                            key=f"field_{column}",
                        )
                    else:
                        st.warning(f"No choices were found for {label}.")
                elif name == "cibilscore":
                    score_default = numeric_medians.get(column, 650.0)
                    if pd.isna(score_default):
                        score_default = 650.0
                    submitted_values[column] = st.number_input(
                        label,
                        min_value=300.0,
                        max_value=900.0,
                        value=max(300.0, min(900.0, float(score_default))),
                        step=10.0,
                        format="%.0f",
                        help=help_text,
                        label_visibility="collapsed",
                        key=f"field_{column}",
                    )
                else:
                    default = numeric_medians.get(column, 0.0)
                    if pd.isna(default):
                        default = 0.0
                    if "income" in name:
                        step = 1000.0
                    elif name in {"loanamount", "existingdebt"}:
                        step = 10000.0
                    elif name == "monthlyinstallment":
                        step = 500.0
                    elif name == "debttoincomeratio":
                        step = 0.01
                    else:
                        step = 1.0
                    submitted_values[column] = st.number_input(
                        label,
                        min_value=0.0,
                        value=float(default),
                        step=step,
                        format="%.2f" if name == "debttoincomeratio" else "%.0f",
                        help=help_text,
                        label_visibility="collapsed",
                        key=f"field_{column}",
                    )

                st.caption(FIELD_CAPTIONS.get(column, f"Enter the {label.lower()} value shown in your CSV."))

        st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Get model prediction", width="stretch")

if submitted:
    application = pd.DataFrame([submitted_values], columns=feature_columns)
    try:
        predicted_class = int(model.predict(application)[0])
        class_names = artifact.get("class_names", {0: "Rejected", 1: "Approved"})
        prediction = str(class_names.get(predicted_class, predicted_class))
        classes = list(model.named_steps["classifier"].classes_)
        approval_probability = None
        if hasattr(model, "predict_proba") and 1 in classes:
            approval_probability = float(model.predict_proba(application)[0][classes.index(1)])

        result_style = "result-approved" if predicted_class == 1 else "result-rejected"
        result_copy = "The model’s predicted class for these details."
        st.markdown(
            f"""
            <div class="result-card {result_style}">
              <p class="result-label">MODEL PREDICTION</p>
              <div class="result-title">{prediction}</div>
              <p class="result-copy">{result_copy}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cibil_column = next(
            (column for column in feature_columns if normalized_name(column) == "cibilscore"),
            None,
        )
        if cibil_column is not None:
            st.caption(f"CIBIL score used by the model: {int(submitted_values[cibil_column])} / 900")

        if approval_probability is not None:
            st.metric("Model-estimated approval probability", f"{approval_probability:.0%}")
            st.progress(approval_probability)

        st.info("This is a classroom prediction from a model trained on fictional sample records. It is not a real loan decision, and the score may not be calibrated.")
    except Exception as exc:
        st.error(f"Could not calculate a prediction: {exc}")
else:
    st.markdown(
        '<p class="fine-print">The form starts with example-like values. Change them, then press <b>Get model prediction</b>. This demo is not for real loan decisions.</p>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("### About this demo")
    st.write("This project demonstrates supervised binary classification for loan application data.")
    st.warning("The included model learned from fictional sample rows. Do not enter real applicant information.")
    st.markdown("### What the result means")
    st.write("The approval probability comes from the model output. It is not the applicant's CIBIL score, a bank decision, or a measure of actual ability to repay.")
