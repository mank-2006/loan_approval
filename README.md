# Loan approval preview

A Streamlit classroom demo that predicts Approved or Rejected from an application profile. The model and records are fictional examples; they are not real lender data and must not be used to make loan decisions.

Applicant and co-applicant income are monthly amounts in Indian rupees (₹). Loan amount is the total requested amount in rupees, the repayment term is in months, and CIBIL score accepts values from 300 to 900.

The classifier also uses an estimated monthly repayment as a share of combined monthly income. The estimate assumes a fixed 12% annual interest rate for demonstration purposes. A high income relative to a small requested loan can improve the estimate, while a high repayment burden or lower CIBIL score can reduce it.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_loan_data.py
python loan_approval_model.py --data loan_data_sample.csv --target Loan_Status
streamlit run app.py
```

The generated CSV contains 4,001 fictional rows, including an example with ₹10 lakh monthly income, a ₹10,000 loan request, and a CIBIL score of 670.

## Data and model limits

The labels are generated from an illustrative classroom policy, not historical loan decisions. The repayment calculation uses a teaching assumption rather than an actual lender rate. The model is for demonstrating a machine-learning workflow only. Do not enter real applicant information in the public website.
