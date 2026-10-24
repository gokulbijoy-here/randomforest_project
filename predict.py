
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

if not MODEL_PATH.exists():
    print("Model not found. Run python train.py first.")
    raise SystemExit(1)

model = joblib.load(MODEL_PATH)

print("\n=== Food Delivery Time Predictor ===")

order = pd.DataFrame([{
    "Distance_km": float(input("Delivery distance (km): ")),
    "Weather": input("Weather (e.g. Clear, Cloudy, Rainy): "),
    "Traffic_Level": input("Traffic (Low, Medium, High): "),
    "Time_of_Day": input("Time (Morning, Afternoon, Evening, Night): "),
    "Vehicle_Type": input("Vehicle (Bike, Scooter, Car): "),
    "Preparation_Time_min": float(
        input("Preparation time (minutes): ")
    ),
    "Courier_Experience_yrs": float(
        input("Courier experience (years): ")
    ),
}])

prediction = model.predict(order)[0]

print(f"\nEstimated delivery time: {prediction:.1f} minutes")
