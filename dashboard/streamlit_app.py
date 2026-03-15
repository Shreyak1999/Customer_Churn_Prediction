import streamlit as st

from models.predict import predict_churn


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Telco Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to churn using a trained ML model.")

# -----------------------------
# Model artifact paths
# -----------------------------
MODEL_PATH = "models/churn_model.pkl"
FEATURE_COLUMNS_PATH = "models/model_features.pkl"

# -----------------------------
# Input form
# -----------------------------
st.subheader("Customer Information")

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.1)
total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0, step=0.1)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Churn"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        result = predict_churn(
            input_data=input_data,
            model_path=MODEL_PATH,
            feature_columns_path=FEATURE_COLUMNS_PATH
        )

        churn_prediction = result["churn_prediction"]
        churn_probability = result["churn_probability"]

        st.subheader("Prediction Result")

        if churn_prediction == 1:
            st.error(f"⚠️ High Risk of Churn")
        else:
            st.success(f"✅ Low Risk of Churn")

        st.metric("Churn Probability", f"{churn_probability * 100:.2f}%")

        if churn_probability >= 0.75:
            st.warning("This customer is at very high churn risk. Retention action recommended.")
        elif churn_probability >= 0.50:
            st.info("This customer has moderate churn risk. Monitor engagement.")
        else:
            st.success("This customer is relatively stable.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")