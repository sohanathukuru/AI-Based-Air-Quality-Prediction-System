# AI-Based Air Quality Prediction System for Personalized Pollution Alerts and Route Optimization

## Team
**Team B**

## Project Overview

Air pollution has become a major public health concern in urban environments. This project aims to predict future Air Quality Index (AQI) values using Machine Learning and Deep Learning techniques while providing a foundation for personalized pollution alerts and cleaner route recommendations.

The system combines comprehensive data preprocessing, feature engineering, Random Forest, and Long Short-Term Memory (LSTM) models to accurately forecast future AQI values from historical pollutant observations.

This project is being developed as part of the **Infosys Springboard Internship 2026**.

---

# Objectives

- Predict future AQI values using historical air quality measurements.
- Compare Machine Learning and Deep Learning approaches.
- Generate reliable AQI forecasts for future integration with route optimization.
- Build a scalable and reproducible air quality prediction pipeline.

---

# Project Workflow

```
Raw CPCB Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Missing Value Handling
        │
        ▼
Feature Engineering
        │
        ▼
Station-wise Chronological Split
        │
        ▼
Random Forest Model
        │
        ├─────────────► Performance Evaluation
        │
        ▼
LSTM Model
        │
        ▼
Future AQI Prediction
        │
        ▼
Personalized Pollution Alerts
        │
        ▼
Route Optimization (Upcoming Phase)
```

---

# Project Structure

```
AI-Based-Air-Quality-Prediction-System
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   ├── random_forest_model.pkl
│   └── lstm_model.keras
│
├── results
│
├── src
│   ├── 01_data_preprocessing.py
│   ├── 02_feature_engineering.py
│   ├── 03_random_forest.py
│   ├── 04_lstm_model.py
│   └── data.py
│
├── tests
│   ├── interpolation_test.py
│   └── kalman_imputation_test.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Dataset

The project uses hourly air quality monitoring data collected from CPCB monitoring stations.

### Cities Included

- Delhi
- Hyderabad

### Features

The dataset contains pollutant measurements including:

- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- Benzene
- Toluene

Additional temporal and engineered features were created to improve prediction performance.

---

# Feature Engineering

The project includes several engineered features including:

### Time Features

- Hour
- Day
- Month
- Year
- Quarter
- Day of Week
- Weekend Indicator

### Lag Features

Examples include:

- AQI Lag 1
- AQI Lag 3
- AQI Lag 6
- AQI Lag 12
- AQI Lag 24

Pollutant lag features were also generated.

### Rolling Statistics

Rolling averages were computed for AQI and important pollutants to capture temporal trends.

---

# Machine Learning Models

## Random Forest Regressor

Used as the baseline Machine Learning model.

### Purpose

- Capture nonlinear relationships
- Estimate feature importance
- Provide a strong regression baseline

---

## Long Short-Term Memory (LSTM)

Used for sequential time-series prediction.

### Architecture

- Two stacked LSTM layers
- Dropout regularization
- Dense hidden layer
- Single output neuron for AQI prediction

The LSTM uses a 24-hour sliding window to learn temporal dependencies in pollutant concentrations.

---

# Data Preparation

The dataset is split using **station-wise chronological splitting**.

Training, validation, and testing data are separated without breaking temporal order to avoid data leakage.

Split ratio:

- Training: 70%- Validation: 15%
- Testing: 15%

---

# Current Milestone Status

Completed:

- Dataset preprocessing
- Missing value handling
- Feature engineering
- Random Forest model training
- LSTM model training
- Model evaluation
- Model serialization
- Project organization



---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Missingno
- Git
- GitHub
- Visual Studio Code

---

# Repository Contents

This repository includes:

- Source code
- Raw datasets
- Processed datasets
- Trained models
- Experimental results
- Test scripts


---

# Team

**Team B**

Infosys Springboard Internship 2026

---

# License


This repository is intended for academic and educational purposes under the Infosys Springboard Internship Program.
