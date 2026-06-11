import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open("trained_model.sav", "rb") as file:
        model = pickle.load(file)
    return model

loaded_model = load_model()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main-header{
    text-align:center;
    padding:1rem;
}

.result-box{
    padding:15px;
    border-radius:10px;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    "<h1 class='main-header'>🩺 Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<center>Machine Learning Based Diabetes Risk Assessment</center>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📊 Model Information")

    st.success("Model Loaded Successfully")

    st.write("**Algorithm:**")
    st.info("Random Forest Classifier")

    st.write("**Dataset:**")
    st.info("Diabetes Dataset")

    st.write("**Input Units:**")

    st.markdown("""
- Glucose → mg/dL
- Blood Pressure → mmHg (Diastolic)
- Skin Thickness → mm
- Insulin → μU/mL
- BMI → kg/m²
- Age → years
""")

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.header("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    Pregnancies = st.number_input(
        "Pregnancies (count)",
        min_value=0,
        value=1,
        help="Number of pregnancies."
    )

    Glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0.0,
        value=120.0,
        help="Plasma glucose concentration."
    )

    BloodPressure = st.number_input(
        "Diastolic Blood Pressure (mmHg)",
        min_value=0.0,
        value=80.0,
        help="Bottom number in BP reading. Example: 120/80 → enter 80."
    )

    SkinThickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0.0,
        value=20.0,
        help="Triceps skin fold thickness."
    )

with col2:

    Insulin = st.number_input(
        "Insulin (μU/mL)",
        min_value=0.0,
        value=80.0,
        help="2-Hour serum insulin."
    )

    st.subheader("BMI Calculator")

    weight = st.number_input(
        "Weight (kg)",
        min_value=1.0,
        value=70.0,
        help="Patient weight in kilograms."
    )

    height = st.number_input(
        "Height (m)",
        min_value=0.5,
        value=1.75,
        help="Patient height in metres."
    )

    if height > 0:
        BMI = weight / (height ** 2)

        st.metric(
            "Calculated BMI",
            f"{BMI:.1f}"
        )

        if BMI < 18.5:
            st.warning("BMI Category: Underweight")
        elif BMI < 25:
            st.success("BMI Category: Normal")
        elif BMI < 30:
            st.warning("BMI Category: Overweight")
        else:
            st.error("BMI Category: Obese")

    else:
        BMI = 25.0

    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.500,
        format="%.3f",
        help="Genetic likelihood score."
    )

    Age = st.number_input(
        "Age (years)",
        min_value=1,
        value=30,
        help="Patient age."
    )
# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button("🔍 Predict Diabetes", use_container_width=True):

    input_data = np.array([
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]).reshape(1, -1)

    prediction = loaded_model.predict(input_data)

    st.divider()

    st.header("📈 Prediction Results")

    # Some models may not support predict_proba
    try:
        probability = loaded_model.predict_proba(input_data)

        diabetic_probability = probability[0][1] * 100
        healthy_probability = probability[0][0] * 100

    except Exception:
        diabetic_probability = 0
        healthy_probability = 0

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    if prediction[0] == 1:

        st.error(
            f"⚠️ High Risk of Diabetes\n\nConfidence: {diabetic_probability:.2f}%"
        )

        st.progress(min(int(diabetic_probability), 100))

    else:

        st.success(
            f"✅ Low Risk of Diabetes\n\nConfidence: {healthy_probability:.2f}%"
        )

        st.progress(min(int(healthy_probability), 100))

    # --------------------------------------------------
    # PATIENT SUMMARY
    # --------------------------------------------------

    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Pregnancies",
            "Glucose",
            "Diastolic BP",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Pedigree Function",
            "Age"
        ],
        "Value": [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ],
        "Unit": [
            "count",
            "mg/dL",
            "mmHg",
            "mm",
            "μU/mL",
            "kg/m²",
            "-",
            "years"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True
    )

    # --------------------------------------------------
    # RISK ANALYSIS
    # --------------------------------------------------

    st.subheader("⚕️ Risk Analysis")

    risks = []

    if Glucose > 140:
        risks.append("High glucose level")

    if BMI > 30:
        risks.append("BMI indicates obesity")

    if BloodPressure > 90:
        risks.append("Elevated diastolic blood pressure")

    if Age > 45:
        risks.append("Age-related risk factor")

    if DiabetesPedigreeFunction > 0.8:
        risks.append("Strong family history indicator")

    if len(risks) == 0:
        st.success("No major risk factors detected.")
    else:
        for risk in risks:
            st.warning(risk)

    # --------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------

    if diabetic_probability > 0:

        st.subheader("📊 Prediction Probabilities")

        chart_data = pd.DataFrame(
            {
                "Probability (%)": [
                    healthy_probability,
                    diabetic_probability
                ]
            },
            index=[
                "Non-Diabetic",
                "Diabetic"
            ]
        )

        st.bar_chart(chart_data)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Diabetes Prediction System | Streamlit + Scikit-Learn + Random Forest"
)