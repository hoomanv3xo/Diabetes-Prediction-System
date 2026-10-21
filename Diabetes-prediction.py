import streamlit as st
import pandas as pd
import altair as alt
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
    st.info("Pima Indians Diabetes (768 patients)")

    st.write("**Performance (5-fold CV):**")
    st.metric("Accuracy", "76.9%")
    st.metric("Recall (diabetic class)", "67.9%")

    st.write("**Input Units:**")

    st.markdown("""
- Glucose → mg/dL
- Blood Pressure → mmHg (Diastolic)
- Skin Thickness → mm
- Insulin → μU/mL
- BMI → kg/m²
- Age → years

Enter **0** for any lab value that is unknown (glucose, blood pressure,
skin thickness, insulin); the model substitutes a typical value.
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

if st.button("🔍 Predict Diabetes", width="stretch"):

    input_data = pd.DataFrame(
        [[
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]],
        columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    )

    probability = loaded_model.predict_proba(input_data)[0]
    healthy_probability = probability[0] * 100
    diabetic_probability = probability[1] * 100

    st.divider()

    st.header("📈 Prediction Results")

    # --------------------------------------------------
    # RESULT (probability bands instead of a hard verdict)
    # --------------------------------------------------

    if diabetic_probability < 35:
        st.success(
            f"✅ Low Risk\n\nEstimated diabetes probability: {diabetic_probability:.1f}%"
        )
    elif diabetic_probability < 60:
        st.warning(
            f"⚠️ Moderate Risk\n\nEstimated diabetes probability: {diabetic_probability:.1f}%"
        )
    else:
        st.error(
            f"🚨 High Risk\n\nEstimated diabetes probability: {diabetic_probability:.1f}%"
        )

    st.progress(min(int(diabetic_probability), 100))

    st.caption(
        "Educational use only. This is not medical advice or a diagnosis. "
        "Consult a qualified healthcare provider."
    )

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
            round(BMI, 1),
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
        width="stretch",
        hide_index=True
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

    st.subheader("📊 Prediction Probabilities")

    chart_df = pd.DataFrame({
        "Outcome": ["Non-Diabetic", "Diabetic"],
        "Probability (%)": [healthy_probability, diabetic_probability],
    })

    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X(
            "Outcome:N",
            sort=["Non-Diabetic", "Diabetic"],
            axis=alt.Axis(labelAngle=0),
            title=None,
        ),
        y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "Outcome:N",
            legend=None,
            scale=alt.Scale(
                domain=["Non-Diabetic", "Diabetic"],
                range=["#2e9e5b", "#d64545"],
            ),
        ),
    )

    st.altair_chart(chart, width="stretch")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Diabetes Prediction System | Streamlit + Scikit-Learn + Random Forest"
)