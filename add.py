import streamlit as st
import pandas as pd
import joblib


# Load saved files

model = joblib.load("loan_approval_model.pkl")

feature_columns = joblib.load("feature_columns.pkl")

scaler = joblib.load("loan_scaler.pkl")


# App Title

st.title("🏦 Loan Approval Prediction System")

st.write(
    "Machine Learning based Loan Eligibility Prediction"
)


# User Inputs

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income"
)

loan_amount = st.number_input(
    "Loan Amount"
)

loan_term = st.number_input(
    "Loan Term"
)

cibil_score = st.number_input(
    "CIBIL Score"
)

residential_assets_value = st.number_input(
    "Residential Assets Value"
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value"
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value"
)

bank_asset_value = st.number_input(
    "Bank Asset Value"
)



# Prediction Button

if st.button("Predict Loan Status"):


    data = pd.DataFrame({

        "no_of_dependents":[no_of_dependents],

        "education":[education],

        "self_employed":[self_employed],

        "income_annum":[income_annum],

        "loan_amount":[loan_amount],

        "loan_term":[loan_term],

        "cibil_score":[cibil_score],

        "residential_assets_value":[residential_assets_value],

        "commercial_assets_value":[commercial_assets_value],

        "luxury_assets_value":[luxury_assets_value],

        "bank_asset_value":[bank_asset_value]

    })


    # Encoding

    data["education"] = data["education"].map(
        {
            "Graduate":0,
            "Not Graduate":1
        }
    )


    data["self_employed"] = data["self_employed"].map(
        {
            "No":0,
            "Yes":1
        }
    )


    # Arrange columns same as training

    data = data[feature_columns]


    # Apply scaler

    data_scaled = scaler.transform(data)


    # Prediction

    prediction = model.predict(data_scaled)


    probability = model.predict_proba(data_scaled)


    # Label mapping

    if prediction[0] == 0:

        result = "✅ Loan Approved"

    else:

        result = "❌ Loan Rejected"


    confidence = round(
        probability[0][prediction[0]] * 100,
        2
    )


    st.success(result)

    st.info(
        f"Confidence: {confidence}%"
    )
