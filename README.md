# Loan Approval Prediction Demo

Supervised machine learning demo that predicts whether a loan application may be approved or rejected from applicant details.

## About this demo

The included model is trained on fictional sample records. It is for learning and demonstration only; it is not a real lending decision tool and should not be used to assess real applicants. The website does not save submitted application details.

## Files

- `app.py`: the interactive Streamlit website.
- `loan_approval_model.joblib`: the trained demo model.
- `requirements.txt`: pinned Python packages for deployment.

## Run locally

Use Python 3.14 to match the model file:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Publish

This repository is prepared for Streamlit Community Cloud. Sign in at [share.streamlit.io](https://share.streamlit.io/), create an app from this repository, select the `main` branch and `app.py`, choose Python 3.14 in advanced settings, and deploy. The resulting `streamlit.app` address can be shared publicly.
