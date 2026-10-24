
# Food Delivery Time Prediction

A beginner-friendly Machine Learning project that predicts
food delivery time using a Random Forest Regressor.

## Features
- Data preprocessing and missing-value handling
- Random Forest regression
- Model evaluation using MAE, RMSE and R²
- Actual vs. predicted visualization
- Prediction for new delivery orders

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Project Structure
food-delivery-time-prediction/
├── data/
│   └── food_delivery.csv
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Place the dataset at data/food_delivery.csv.
5. Train the model:

   python train.py

6. Predict delivery time:

   python predict.py

## Evaluation
The model reports MAE, RMSE and R² on a held-out test set.

## Dataset
Add the dataset source and license here.

## Disclaimer
Predictions are estimates and may not reflect actual delivery times.
