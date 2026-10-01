
import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="centered"
)


# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("model_joblib.pkl")


# =====================================================
# TITLE
# =====================================================

st.title("💰 Loan Approval Prediction")

st.write(
    "Enter the applicant details below to predict "
    "whether the loan will be approved or rejected."
)


# =====================================================
# INPUT DETAILS
# =====================================================

st.header("👤 Applicant Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Widowed"]
)

education_level = st.selectbox(
    "Education Level",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

annual_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=4000.0
)

employment_status = st.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)

debt_to_income_ratio = st.number_input(
    "Debt-to-Income Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.01
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300.0,
    max_value=900.0,
    value=700.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Personal", "Home", "Education", "Car", "Business"]
)

interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    value=10.0
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    max_value=120,
    value=36
)

installment = st.number_input(
    "Installment",
    min_value=0.0,
    value=300.0
)


# =====================================================
# CREDIT DETAILS
# =====================================================

st.header("📊 Credit Details")

grade_subgrade = st.text_input(
    "Grade / Subgrade",
    value="B2"
)

num_of_open_accounts = st.number_input(
    "Number of Open Accounts",
    min_value=0,
    value=5
)

total_credit_limit = st.number_input(
    "Total Credit Limit",
    min_value=0.0,
    value=50000.0
)

current_balance = st.number_input(
    "Current Balance",
    min_value=0.0,
    value=10000.0
)

delinquency_history = st.number_input(
    "Delinquency History",
    min_value=0,
    value=0
)

public_records = st.number_input(
    "Public Records",
    min_value=0,
    value=0
)

num_of_delinquencies = st.number_input(
    "Number of Delinquencies",
    min_value=0,
    value=0
)


# =====================================================
# PREDICTION
# =====================================================

if st.button("🔍 Predict Loan Approval"):

    # Create input DataFrame
    input_data = pd.DataFrame([{

        "age": age,

        "gender": gender,

        "marital_status": marital_status,

        "education_level": education_level,

        "annual_income": annual_income,

        "monthly_income": monthly_income,

        "employment_status": employment_status,

        "debt_to_income_ratio": debt_to_income_ratio,

        "credit_score": credit_score,

        "loan_amount": loan_amount,

        "loan_purpose": loan_purpose,

        "interest_rate": interest_rate,

        "loan_term": loan_term,

        "installment": installment,

        "grade_subgrade": grade_subgrade,

        "num_of_open_accounts": num_of_open_accounts,

        "total_credit_limit": total_credit_limit,

        "current_balance": current_balance,

        "delinquency_history": delinquency_history,

        "public_records": public_records,

        "num_of_delinquencies": num_of_delinquencies

    }])


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Display result
    st.subheader("📋 Prediction Result")


    if prediction == 1:

        st.success("✅ Loan Approved")

    else:

        st.error("❌ Loan Rejected")


    # Probability if model supports it
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0][1]

        st.write(
            f"Approval Probability: "
            f"**{probability * 100:.1f}%**"
        )

        st.progress(float(probability))

