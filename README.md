# Loan approval preview

A Streamlit classroom demo that predicts Approved or Rejected using the included model. The model and sample records are fictional and are not suitable for real lending decisions.

Applicant and co-applicant income fields are monthly amounts in Indian rupees (₹). Other money fields are also in rupees, and loan amount means the total amount requested. The CIBIL score field accepts values from 300 to 900.

## Run locally

Use Python 3.14 to match the saved model:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Publish with Streamlit Community Cloud

1. Upload the files in this folder to a GitHub repository.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Choose **Create app**, select the repository and branch, and set the app file to `app.py`.
4. In advanced settings, select Python 3.14 to match the saved model.
5. Deploy the app and share the generated `streamlit.app` address.

Use only fictional data in this public demo. The model is trained on synthetic records and should not be used to make real loan decisions.
