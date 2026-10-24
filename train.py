
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "food_delivery.csv"
MODEL_PATH = BASE_DIR / "model.joblib"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

# 2. Select features and target
features = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

target = "Delivery_Time_min"

required_columns = features + [target]
missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

# 3. Clean data
df = df[required_columns].copy()

numeric_columns = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    target,
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[target])
df = df[df[target] > 0]

print("Dataset shape after cleaning:", df.shape)

X = df[features]
y = df[target]

# 4. Identify feature types
numeric_features = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

categorical_features = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]

# 5. Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 6. Build Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )),
])

# 7. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 8. Train model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# 9. Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"MAE:  {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print(f"R² score: {r2:.3f}")

# 10. Plot actual vs predicted delivery time
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.6)

lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())

plt.plot([lower, upper], [lower, upper], "r--")
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "actual_vs_predicted.png",
    dpi=150,
)
plt.close()

# 11. Save trained pipeline
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:", MODEL_PATH)
print("Graph saved: actual_vs_predicted.png")
print("\nTraining completed successfully!")
