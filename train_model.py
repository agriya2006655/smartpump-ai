"""
train_model.py
---------------
Loads pump_data.csv, trains a Random Forest to classify pump condition,
prints accuracy + which sensors matter most, and saves the trained
model to pump_model.pkl so app.py can reuse it instantly (no retraining
every time you open the dashboard).

Run this AFTER generate_data.py.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data
df = pd.read_csv("pump_data.csv")

FEATURES = [
    "Temperature_C", "Vibration_mm_s", "Inlet_Pressure_bar",
    "Outlet_Pressure_bar", "Flow_Rate_Lmin", "RPM", "Power_kW",
    "Operating_Hours"
]
TARGET = "Condition"

X = df[FEATURES]
y = df[TARGET]

# 2. Split into train/test so we can honestly check performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train the Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# 4. Check how good it is
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.1f}%\n")
print(classification_report(y_test, y_pred))

# 5. See which sensors matter most (this powers the "AI Diagnosis" factors)
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
print("Feature importance (which sensors drive the prediction most):")
print(importances)

# 6. Save the trained model to disk so the dashboard can load it instantly
joblib.dump(model, "pump_model.pkl")
print("\nSaved trained model to pump_model.pkl")
