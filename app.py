"""Local Streamlit interface for the loan approval model demo."""

from pathlib import Path
import re

import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("loan_approval_model.joblib")

FEATURE_LABELS = {
    "ApplicantIncome": "Applicant income",
    "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Requested loan amount",
    "Loan_Amount_Term": "Loan term (months)",
    "Credit_History": "Credit history code",
    "Employment_Status": "Employment status",
    "Self_Employed": "Self-employed status",
    "Property_Area": "Property area",
    "Monthly_Installment": "Monthly installment",
    "Existing_Debt": "Existing debt",
    "Debt_To_Income_Ratio": "Debt-to-income ratio",
}


def pretty_name(column: str) -> str:
    readable = re.sub(r"(?<!^)(?=[A-Z])", " ", column.replace("_", " "))
    return FEATURE_LABELS.get(column, readable)


def to_python(value):
    """Convert numpy scalar values from the saved encoder to Python values."""
    return value.item() if hasattr(value, "item") else value


st.set_page_config(
    page_title="Loan approval preview",
    page_icon="🏦",
    layout="centered",
)

st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(180deg, #f2f7f4 0%, #fbfcfb 42%, #ffffff 100%);
      }
      .block-container { max-width: 900px; padding-top: 2.4rem; padding-bottom: 3rem; }
      .hero {
        padding: 1.7rem 1.9rem 1.6rem;
        border: 1px solid #dce9df;
        border-radius: 22px;
        background: rgba(255,255,255,.92);
        box-shadow: 0 14px 42px rgba(20, 58, 39, .07);
        margin-bottom: 1.25rem;
      }
      .eyebrow {
        color: #23744b; font-size: .76rem; font-weight: 750;
        letter-spacing: .14em; text-transform: uppercase; margin-bottom: .55rem;
      }
      .hero h1 { color: #163d2b; font-size: 2.15rem; margin: 0; letter-spacing: -.04em; }
      .hero p { color: #5d7064; font-size: 1.02rem; margin: .55rem 0 0; }
      div[data-testid="stForm"] {
        background: #fff; border: 1px solid #e2ebe5; border-radius: 18px;
        padding: 1.15rem 1.25rem 1rem;
      }
      div.stButton > button, div[data-testid="stFormSubmitButton"] button {
        background: #176b43; color: #fff; border: 0; border-radius: 10px;
        min-height: 2.9rem; font-weight: 650;
      }
      div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: #105634; color: #fff; border: 0;
      }
      .footnote { color: #718078; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI for Engineers · MST project</div>
      <h1>Loan approval preview</h1>
      <p>Interactive classroom demo trained on fictional sample records.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.warning("The trained model file is not here yet.")
    st.markdown(
        "Train the model first, then refresh this page. From this folder, run:"
    )
    st.code(
        "python loan_approval_model.py --data loan_data_sample.csv --target Loan_Status",
        language="bash",
    )
    st.stop()

try:
    import joblib

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["features"]
except Exception as exc:
    st.error(f"Could not load the model file: {exc}")
    st.stop()

preprocessor = model.named_steps["preprocessor"]
numeric_columns = list(preprocessor.transformers_[0][2])
categorical_columns = list(preprocessor.transformers_[1][2])

numeric_pipeline = preprocessor.named_transformers_["numeric"]
numeric_medians = dict(
    zip(numeric_columns, numeric_pipeline.named_steps["imputer"].statistics_)
)

categorical_pipeline = preprocessor.named_transformers_["categorical"]
encoder = categorical_pipeline.named_steps["onehot"]
categorical_choices = {
    column: [to_python(value) for value in categories]
    for column, categories in zip(categorical_columns, encoder.categories_)
}

st.subheader("Application details")
st.caption("Enter values in the same units and formats used in the training CSV.")

with st.form("loan_application_form"):
    left, right = st.columns(2, gap="large")
    submitted_values = {}

    for index, column in enumerate(feature_columns):
        container = left if index % 2 == 0 else right
        label = pretty_name(column)

        with container:
            if column in categorical_columns:
                choices = categorical_choices[column]
                if not choices:
                    st.error(f"No training categories were found for {label}.")
                    st.stop()
                submitted_values[column] = st.selectbox(
                    label,
                    options=choices,
                    format_func=str,
                    help="Choices come from the categories in the training data.",
                )
            elif column.lower().replace("_", "") == "credithistory":
                submitted_values[column] = st.selectbox(
                    label,
                    options=[1.0, 0.0],
                    format_func=lambda value: str(int(value)),
                    help="Use the meaning assigned to 1 and 0 in your dataset.",
                )
            else:
                default = numeric_medians.get(column, 0.0)
                if pd.isna(default):
                    default = 0.0
                step = 1000.0 if "income" in column.lower() else 1.0
                submitted_values[column] = st.number_input(
                    label,
                    min_value=0.0,
                    value=float(default),
                    step=step,
                    help="Use the same units as the values in your training CSV.",
                )

    submitted = st.form_submit_button("Check model prediction", width="stretch")

if submitted:
    application = pd.DataFrame([submitted_values], columns=feature_columns)
    predicted_class = int(model.predict(application)[0])
    class_names = artifact.get("class_names", {0: "Rejected", 1: "Approved"})
    prediction = class_names.get(predicted_class, str(predicted_class))

    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    approval_probability = None
    if hasattr(model, "predict_proba") and 1 in classes:
        approval_probability = float(model.predict_proba(application)[0][classes.index(1)])

    st.divider()
    st.subheader("Model result")
    if predicted_class == 1:
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Prediction: {prediction}")

    if approval_probability is not None:
        st.metric("Estimated approval probability", f"{approval_probability:.1%}")
        st.progress(approval_probability)

    st.markdown(
        '<p class="footnote">This is an educational demo trained on fictional sample records. '
        "It is not a real lending decision, and the displayed probability may not be calibrated.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<p class="footnote">Your entered values are used for this prediction and are not saved by this app.</p>',
        unsafe_allow_html=True,
    )
