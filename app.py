import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and metadata
model = joblib.load("salary_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # All columns used during training

# Streamlit UI
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction")

# Inputs
gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
education = st.selectbox("Education Level", label_encoders["Education Level"].classes_)
job_title = st.selectbox("Job Title", sorted([col.replace("Job Title_", "") for col in feature_columns if col.startswith("Job Title_")]))
age = st.number_input("Age", min_value=18, max_value=70, value=25)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)

# Encode inputs
try:
    gender_encoded = label_encoders["Gender"].transform([gender])[0]
    education_encoded = label_encoders["Education Level"].transform([education])[0]

    # Build initial input dictionary
    input_data = {
        "Gender": gender_encoded,
        "Education Level": education_encoded,
        "Age": age,
        "Years of Experience": experience
    }

    # One-hot encode job title
    for col in feature_columns:
        if col.startswith("Job Title_"):
            input_data[col] = 1 if col == f"Job Title_{job_title}" else 0

    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    if st.button("🔍 Predict Salary"):
        annual_salary = model.predict(input_df)[0]
        monthly_salary = annual_salary / 12

        st.success(f"💰 Predicted Monthly Salary: ${monthly_salary:,.2f}")
        st.info(f"📅 Predicted Annual Salary: ${annual_salary:,.2f}")
except Exception as e:
    st.error(f"Prediction failed: {e}")
