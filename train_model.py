import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix)

# Columns where 0 means "not recorded" (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
MISSING_AS_ZERO = [1, 2, 3, 4, 5]


def build_model():
    prep = ColumnTransformer(
        [("impute_zeros", SimpleImputer(missing_values=0, strategy="median"), MISSING_AS_ZERO)],
        remainder="passthrough",
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",   # improves recall on the minority (diabetic) class
        random_state=42,
    )
    return Pipeline([("prep", prep), ("rf", rf)])


df = pd.read_csv("diabetes.csv")
print("Rows:", len(df))
X, y = df.drop("Outcome", axis=1), df["Outcome"]

# Hold-out evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = build_model().fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Hold-out  Accuracy : {accuracy_score(y_test, pred):.1%}")
print(f"Hold-out  Recall   : {recall_score(y_test, pred):.1%}")
print(f"Hold-out  Precision: {precision_score(y_test, pred):.1%}")
print(f"Hold-out  F1       : {f1_score(y_test, pred):.1%}")
print(confusion_matrix(y_test, pred))

# 5-fold cross-validation (more reliable on a small dataset)
cv = cross_validate(
    build_model(), X, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring=["accuracy", "recall", "precision", "f1"],
)
for m in ["accuracy", "recall", "precision", "f1"]:
    print(f"5-fold CV {m:<9}: {cv['test_' + m].mean():.1%}")

# Final model: train on all data and save
final_model = build_model().fit(X, y)
with open("trained_model.sav", "wb") as f:
    pickle.dump(final_model, f)
print("trained_model.sav saved")
