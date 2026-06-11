import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA & TRAIN MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    df = pd.read_csv("diabetes.csv")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy, df


model, accuracy, df = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🩺 Diabetes Prediction System")
st.markdown("Machine Learning Based Diabetes Risk Assessment")

st.divider()

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.header("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Patient Records",
        len(df)
    )

with col2:
    st.metric(
        "Features",
        len(df.columns) - 1
    )

with col3:
    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

st.success(
    f"Model trained successfully using {len(df)} records from diabetes.csv"
)

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

with st.expander("📄 View Dataset Preview"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# --------------------------------------------------
# DATASET STATISTICS
# --------------------------------------------------

with st.expander("📈 Dataset Statistics"):

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# --------------------------------------------------
# OUTCOME DISTRIBUTION
# --------------------------------------------------

st.subheader("Outcome Distribution")

outcome_counts = df["Outcome"].value_counts()

st.bar_chart(outcome_counts)

st.divider()

# --------------------------------------------------
# PATIENT INPUT
# --------------------------------------------------

st.header("👤 Patient Information")

left, right = st.columns(2)

with left:

    Pregnancies = st.number_input(
        "Pregnancies (count)",
        min_value=0,
        value=1
    )

    Glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0.0,
        value=120.0
    )

    BloodPressure = st.number_input(
        "Diastolic Blood Pressure (mmHg)",
        min_value=0.0,
        value=80.0
    )

    SkinThickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0.0,
        value=20.0
    )

with right:

    Insulin = st.number_input(
        "Insulin (μU/mL)",
        min_value=0.0,
        value=80.0
    )

    st.subheader("BMI Calculator")

    weight = st.number_input(
        "Weight (kg)",
        min_value=1.0,
        value=70.0
    )

    height = st.number_input(
        "Height (m)",
        min_value=0.5,
        value=1.75
    )

    BMI = weight / (height ** 2)

    st.metric(
        "Calculated BMI",
        f"{BMI:.1f}"
    )

    if BMI < 18.5:
        st.warning("BMI Category: Underweight")
    elif BMI < 25:
        st.success("BMI Category: Normal Weight")
    elif BMI < 30:
        st.warning("BMI Category: Overweight")
    else:
        st.error("BMI Category: Obese")

    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.5,
        format="%.3f"
    )

    Age = st.number_input(
        "Age (years)",
        min_value=1,
        value=30
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
):

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

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    diabetic_probability = probability[0][1] * 100
    healthy_probability = probability[0][0] * 100

    st.divider()

    st.header("📋 Prediction Results")

    if prediction[0] == 1:

        st.error(
            f"⚠️ High Risk of Diabetes\n\nConfidence: {diabetic_probability:.2f}%"
        )

        st.progress(int(diabetic_probability))

    else:

        st.success(
            f"✅ Low Risk of Diabetes\n\nConfidence: {healthy_probability:.2f}%"
        )

        st.progress(int(healthy_probability))

    # ----------------------------------------------
    # RECOMMENDATIONS
    # ----------------------------------------------

    st.subheader("💡 Health Recommendations")

    if BMI > 30:
        st.info(
            "Consider weight management strategies and regular physical activity."
        )

    if Glucose > 140:
        st.info(
            "Discuss glucose monitoring and diabetes screening with a healthcare professional."
        )

    if BloodPressure > 90:
        st.info(
            "Monitor blood pressure regularly and maintain a healthy lifestyle."
        )

    if Age > 45:
        st.info(
            "Regular health checkups may help identify risk factors early."
        )

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()

st.header("📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": df.drop("Outcome", axis=1).columns,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

st.dataframe(
    importance_df,
    use_container_width=True
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Diabetes Prediction System | Streamlit + Random Forest + BMI Calculator"
)

