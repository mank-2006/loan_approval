# Loan Approval Prediction Demo

A supervised classification project that predicts an application as Approved or Rejected from applicant details.

## Website

The interactive Streamlit app is in `app.py`. The trained model artifact is `loan_approval_model.joblib`.

The applicant and co-applicant income fields are monthly income in Indian rupees (₹). Other money fields use rupees, and loan amount means the total amount requested.

## Data and limitations

The included sample rows and trained model are fictional and intended for classroom demonstration only. Do not use the results to make real lending decisions or enter real applicant information.

## Run locally

Use Python 3.14 to match the saved model:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy

The app can be deployed on [Streamlit Community Cloud](https://share.streamlit.io/). Create an app from this repository, select the `main` branch and `app.py`, choose Python 3.14 in advanced settings, then deploy.
