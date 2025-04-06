import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Generate a complete dataset with realistic values
np.random.seed(42)

# Define the possible values for each feature
num_samples = 10000  # Increase this for a larger dataset

data = {
    "Race": np.random.randint(0, 3, num_samples),  # 3 Races (0, 1, 2)
    "Location_Risk": np.random.randint(0, 3, num_samples),  # Low, Medium, High (0, 1, 2)
    "Suspect_Profile": np.random.randint(0, 3, num_samples),  # No Suspicion, Suspicious, Known Threat
    "Weapon_Detected": np.random.randint(0, 2, num_samples),  # 0 = No, 1 = Yes
    "Clothing_Color": np.random.randint(0, 3, num_samples),  # Red, Black, White
    "Past_Criminal_Record": np.random.randint(0, 2, num_samples),  # 0 = No, 1 = Yes
    "Nervous_Behavior": np.random.randint(0, 2, num_samples),  # 0 = No, 1 = Yes
    "Gang_Affiliation": np.random.randint(0, 2, num_samples),  # 0 = No, 1 = Yes
    "Time_of_Day": np.random.randint(0, 3, num_samples),  # Morning, Evening, Night
    "Carrying_Bag": np.random.randint(0, 2, num_samples),  # 0 = No, 1 = Yes
    "Eye_Contact": np.random.randint(0, 2, num_samples),  # 0 = Avoids, 1 = Maintains
    "Running": np.random.randint(0, 2, num_samples),  # 0 = No, 1 = Yes
}

# Convert to DataFrame
df = pd.DataFrame(data)

# ✅ Define threat classification rules
df["Threat"] = (
    (df["Weapon_Detected"] == 1) |  # If a weapon is detected → Higher chance of threat
    ((df["Suspect_Profile"] == 2) & (df["Location_Risk"] == 2)) |  # Known threat in High-Risk area
    ((df["Past_Criminal_Record"] == 1) & (df["Nervous_Behavior"] == 1)) |  # Criminal record + nervous behavior
    ((df["Gang_Affiliation"] == 1) & (df["Running"] == 1))  # Gang affiliation + running away
).astype(int)

# ✅ Train the Model
X = df.drop(columns=["Threat"])
y = df["Threat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use a robust classifier with better generalization
model = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Save the model in a format compatible with scikit-learn 1.5.2
model_path = "static/threat_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved as '{model_path}' (Compatible with scikit-learn 1.5.2)")
