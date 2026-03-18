# app.py
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved pipeline
# -----------------------------
pipeline = joblib.load("risk_pipeline_5inputs.joblib")

st.set_page_config(page_title="Financial Risk Profiling", layout="centered")
st.title("💰 Financial Risk Profiling System")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Customer Financial Details")

loan_amnt = st.sidebar.number_input("Loan Amount", min_value=1000, max_value=1000000, value=50000, step=1000)
term = st.sidebar.selectbox("Loan Term (Months)", [36, 60])
annual_inc = st.sidebar.number_input("Annual Income", min_value=10000, max_value=5000000, value=500000, step=5000)
dti = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 15.0)
fico = st.sidebar.slider("FICO Score", 300, 850, 700)

# -----------------------------
# Prepare Input for Pipeline
# -----------------------------
input_df = pd.DataFrame({
    'loan_amnt': [loan_amnt],
    'term': [term],
    'annual_inc': [annual_inc],
    'dti': [dti],
    'fico_range_low': [fico]
})



# -----------------------------
# Predict Risk
# -----------------------------
if st.button("Assess Risk"):
    prob = pipeline.predict_proba(input_df)[0][1]  # Probability of being high risk
    score = int(prob * 100)


    # Risk Category & Recommendation
    if prob < 0.3:
        category = "LOW RISK"
        if loan_amnt < 50000:
            rec = "Equity / Mutual Funds / Small Personal Loan"
        else:
            rec = "Equity / Mutual Funds / Home Loan Eligible"
    elif prob < 0.6:
        category = "MEDIUM RISK"
        if dti > 20:
            rec = "Balanced Funds / Secured Loan / Co-signer Required"
        else:
            rec = "Balanced Funds / Moderate Personal Loan"
    else:
        category = "HIGH RISK"
        if fico < 650:
            rec = "Fixed Deposit / Bonds / Avoid Large Loans"
        else:
            rec = "Fixed Deposit / Bonds / Small Secured Loan Only"

    # Dynamic Reasoning
    reasons = []
    if loan_amnt > 200000:
        reasons.append("Large loan amount")
    if term == 60:
        reasons.append("Long loan term")
    if dti > 25:
        reasons.append("High debt-to-income ratio")
    if fico < 660:
        reasons.append("Low credit score")
    if annual_inc < 400000:
        reasons.append("Low annual income")

    if not reasons:
        reason_text = "Customer has favorable financial profile."
    else:
        reason_text = " + ".join(reasons) + " → higher default risk."

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("🔹 Risk Assessment Results")
    st.metric("Risk Probability", f"{prob:.2%}")
    st.metric("Risk Score", f"{score}/100")
    st.success(f"Risk Category: {category}")
    st.info(f"Recommended Product: {rec}")
    st.warning(f"Reason: {reason_text}")


