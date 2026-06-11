
#🩺 Diabetes Prediction System
A machine learning web application that predicts the risk of diabetes based on patient health metrics. Built with Streamlit and a Random Forest Classifier trained on the Pima Indians Diabetes dataset.
Features
Interactive web UI with real-time BMI calculator
Instant diabetes risk prediction with confidence scores
Visual probability chart (Diabetic vs Non-Diabetic)
Automatic risk factor analysis (glucose, BMI, blood pressure, age, family history)
Patient summary table for all entered values
Project Structure
```
├── Diabetes-prediction.py   # Streamlit web app
├── train_model.py           # Model training script
├── trained_model.sav        # Pre-trained Random Forest model (pickle)
└── diabetes.csv             # Training dataset
```
Prerequisites
Python 3.8+
Installation
Clone the repository
```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
Install dependencies
```bash
   pip install streamlit scikit-learn pandas numpy
   ```
Running the App
The `trained_model.sav` file is included, so you can run the app directly without retraining:
```bash
streamlit run Diabetes-prediction.py
```
Then open your browser at `http://localhost:8501`.
Retraining the Model
To retrain the model on updated data, run:
```bash
python train_model.py
```
This will print the model accuracy and overwrite `trained_model.sav`.
Input Features
Feature	Description	Unit
Pregnancies	Number of pregnancies	count
Glucose	Plasma glucose concentration	mg/dL
Blood Pressure	Diastolic blood pressure	mmHg
Skin Thickness	Triceps skin fold thickness	mm
Insulin	2-hour serum insulin	μU/mL
BMI	Body mass index (auto-calculated)	kg/m²
Diabetes Pedigree Function	Genetic likelihood score	—
Age	Patient age	years
Model Details
Property	Value
Algorithm	Random Forest Classifier
Estimators	200 trees
Train/Test Split	80% / 20%
Random State	42
Output	Binary (0 = Non-Diabetic, 1 = Diabetic)
Risk Factors Detected
The app automatically flags the following risk indicators:
Glucose > 140 mg/dL
BMI > 30 (obese)
Diastolic blood pressure > 90 mmHg
Age > 45
Diabetes Pedigree Function > 0.8
Dependencies
Package	Purpose
`streamlit`	Web application framework
`scikit-learn`	Model training and prediction
`pandas`	Data handling and display
`numpy`	Numerical input processing
Disclaimer
This tool is intended for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any health concerns.
