# Real Estate Price Prediction App

This Streamlit app predicts house prices based on various property features using a Linear Regression model.

Developed as part of the CST2216 Individual Term Project at Algonquin College.

---

## Features

- Trains a Linear Regression model on startup
- Interactive Streamlit UI for real-time predictions
- Model performance metrics (MAE, RMSE, R²)
- Modular codebase for maintainability
- Logging and exception handling

---

## Project Structure

real_estate_app/
├── app.py
├── main.py
├── data/
│ └── final.csv
├── config/
│ └── logger.py
├── models/
│ └── model.py
├── preprocessing/
│ └── preprocessing.py
├── utils/
│ └── helpers.py
└── README.md