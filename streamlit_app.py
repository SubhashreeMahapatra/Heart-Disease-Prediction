import os
project_root = os.path.dirname(os.path.abspath(_file_))
preprocessor_path = os.path.join(project_root, 'Datalog', 'preprocessor.pkl')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from source.pipeline.prediction_pipeline import PredictPipeline, CustomData

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("💓 Heart Disease Risk Predictor")
st.markdown("Enter patient details below to assess the risk of heart disease.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
    gender = st.selectbox("Gender", ["Male", "Female"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Prepare custom data
    input_data = CustomData(
        age=age,
        systolic_bp=systolic_bp,
        cholesterol=cholesterol,
        gender=gender,
        diabetes=diabetes,
        smoking_status=smoking_status,
        chest_pain_type=chest_pain_type
    )

    df = input_data.get_data_as_dataframe()

    try:
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)

        result = "Positive Risk (Heart Disease Detected)" if prediction[0] == 1 else "Negative Risk (No Heart Disease)"
        color = "red" if prediction[0] == 1 else "green"

        st.markdown(f"<h3 style='color:{color};'>{result}</h3>", unsafe_allow_html=True)

        # Display patient data
        st.subheader("🧾 Patient Input Chart")
        st.dataframe(df)

        # Show graphs and insights
        st.subheader("📊 Patient Risk Insight Chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = pd.Series([age, systolic_bp, cholesterol], index=["Age", "BP", "Cholesterol"])
        bars.plot(kind='bar', color=['skyblue', 'orange', 'lightcoral'], ax=ax)
        ax.set_ylabel("Values")
        st.pyplot(fig)

        st.subheader("🧠 Medical Insights")
        st.markdown(f"""
        - Age: Higher age increases risk. Patient is {age} years old.
        - Blood Pressure: Systolic BP is {systolic_bp} mmHg. Risk rises above 130 mmHg.
        - Cholesterol: Current value is {cholesterol} mg/dL.
        - Smoking: Smoking status is **{smoking_status}.
        - Diabetes: Diabetes status is **{diabetes}.
        - Chest Pain Type: {chest_pain_type}
        """)
        
        st.info("⚠ Always consult a cardiologist for final diagnosis. This app gives a probability-based prediction only.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
