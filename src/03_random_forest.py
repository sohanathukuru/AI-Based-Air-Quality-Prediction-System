import numpy as np 
import pandas as pd 
# ============================================================
# Load Feature Engineered Dataset
# ============================================================

aqi_df = pd.read_csv("/Users/sohan/Documents/AI-Based-Air-Quality-Prediction-System/data/processed/feature_engineered_dataset.csv")

aqi_df["Datetime"] = pd.to_datetime(aqi_df["Datetime"])

print("\n")
print("=" * 80)
print("PHASE 18.1 : MODEL DATA PREPARATION")
print("=" * 80)

print(f"Dataset Shape : {aqi_df.shape}")




feature_columns = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",

    "Hour",
    "Day",
    "Month",
    "Year",
    "DayOfWeek",
    "IsWeekend",
    "Quarter",

    "AQI_lag_1",
    "AQI_lag_3",
    "AQI_lag_6",
    "AQI_lag_12",
    "AQI_lag_24",

    "PM2.5_lag_1",
    "PM2.5_lag_3",
    "PM10_lag_1",
    "NO2_lag_1",
    "CO_lag_1",

    "AQI_roll_mean_3",
    "AQI_roll_mean_6",
    "AQI_roll_mean_12",

    "PM2.5_roll_mean_3",
    "PM2.5_roll_mean_6",

    "PM10_roll_mean_3",

    "NO2_roll_mean_3",

    "CO_roll_mean_3"
]

target_column = "AQI"


# ============================================================
# Create Model Dataset
#
# StationId and Datetime are retained ONLY for
# station-wise chronological splitting.
# They will NOT be used as model features.
# ============================================================

model_df = aqi_df[
    ["StationId", "Datetime"] +
    feature_columns +
    [target_column]
].dropna().copy()

print("\n")
print("=" * 80)
print("MODEL DATASET SUMMARY")
print("=" * 80)

print(f"Rows Available : {len(model_df):,}")
print(f"Features       : {len(feature_columns)}")
print(f"Target         : {target_column}")


X = model_df[feature_columns]

y = model_df[target_column]

print("\n")
print("=" * 80)
print("FEATURE MATRIX")
print("=" * 80)

print("X Shape :", X.shape)
print("y Shape :", y.shape)


# ============================================================
# Sort Before Splitting
# ============================================================

model_df.sort_values(
    by=["StationId", "Datetime"],
    inplace=True
)

model_df.reset_index(
    drop=True,
    inplace=True
)



# ============================================================
# Phase 18.2 : Station-wise Chronological Split
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 18.2 : STATION-WISE CHRONOLOGICAL SPLIT")
print("=" * 80)

train_list = []
validation_list = []
test_list = []

# ------------------------------------------------------------
# Split Each Station Independently
# ------------------------------------------------------------

for station, station_df in model_df.groupby("StationId"):

    station_df = station_df.sort_values("Datetime")

    total_rows = len(station_df)

    train_end = int(total_rows * 0.70)
    validation_end = int(total_rows * 0.85)

    train_list.append(
        station_df.iloc[:train_end]
    )

    validation_list.append(
        station_df.iloc[
            train_end:validation_end
        ]
    )

    test_list.append(
        station_df.iloc[
            validation_end:
        ]
    )

# ------------------------------------------------------------
# Merge All Stations
# ------------------------------------------------------------

train_df = pd.concat(
    train_list,
    ignore_index=True
)

validation_df = pd.concat(
    validation_list,
    ignore_index=True
)

test_df = pd.concat(
    test_list,
    ignore_index=True
)

print("\n")
print("=" * 80)
print("SPLIT SUMMARY")
print("=" * 80)

print(f"Training Rows   : {len(train_df):,}")
print(f"Validation Rows : {len(validation_df):,}")
print(f"Testing Rows    : {len(test_df):,}")

print("\nStations Included")

print(f"Training   : {train_df['StationId'].nunique()}")
print(f"Validation : {validation_df['StationId'].nunique()}")
print(f"Testing    : {test_df['StationId'].nunique()}")

print("=" * 80)

print("\n")
print("=" * 80)
print("DATE RANGE VALIDATION")
print("=" * 80)

print(f"Training   : {train_df['Datetime'].min()}  >  {train_df['Datetime'].max()}")
print(f"Validation : {validation_df['Datetime'].min()}  >  {validation_df['Datetime'].max()}")
print(f"Testing    : {test_df['Datetime'].min()}  >  {test_df['Datetime'].max()}")

print("=" * 80)






# ============================================================
# Prepare Final Machine Learning Datasets
# ============================================================

X_train = train_df[feature_columns]

y_train = train_df[target_column]

X_validation = validation_df[feature_columns]

y_validation = validation_df[target_column]

X_test = test_df[feature_columns]

y_test = test_df[target_column]

print("\n")
print("=" * 80)
print("FINAL DATASET SHAPES")
print("=" * 80)

print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")

print()

print(f"X_validation : {X_validation.shape}")
print(f"y_validation : {y_validation.shape}")

print()

print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")

print("=" * 80)



# ============================================================
# Phase 19.1 : Import Machine Learning Libraries
# ============================================================

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

print("\n")
print("=" * 80)
print("PHASE 19.1 : RANDOM FOREST IMPORTS")
print("=" * 80)

print("Random Forest Libraries Imported Successfully")

print("=" * 80)




# ============================================================
# Phase 19.2 : Initialize Random Forest
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 19.2 : RANDOM FOREST INITIALIZATION")
print("=" * 80)

rf_model = RandomForestRegressor(

    n_estimators=200,

    max_depth=20,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)

print(rf_model)

print("=" * 80)



# ============================================================
# Phase 19.3 : Train Random Forest
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 19.3 : RANDOM FOREST TRAINING")
print("=" * 80)

rf_model.fit(
    X_train,
    y_train
)

print("Random Forest Training Completed Successfully")

print("=" * 80)



# ============================================================
# Phase 19.4 : Validation Prediction
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 19.4 : VALIDATION PREDICTION")
print("=" * 80)

y_validation_pred = rf_model.predict(X_validation)

print("Validation Prediction Completed Successfully")

print(f"Validation Predictions : {len(y_validation_pred):,}")

print("=" * 80)


# ============================================================
# Phase 19.5 : TEST PREDICTION
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 19.5 : TEST PREDICTION")
print("=" * 80)

y_test_pred = rf_model.predict(X_test)

print("Test Prediction Completed Successfully")

print(f"Test Predictions : {len(y_test_pred):,}")

print("=" * 80)



# ============================================================
# Phase 19.6 : Random Forest Evaluation
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 19.6 : RANDOM FOREST EVALUATION")
print("=" * 80)

validation_mae = mean_absolute_error(
    y_validation,
    y_validation_pred
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        y_validation_pred
    )
)

validation_r2 = r2_score(
    y_validation,
    y_validation_pred
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

print("\nValidation Performance")
print("-" * 40)

print(f"MAE  : {validation_mae:.4f}")
print(f"RMSE : {validation_rmse:.4f}")
print(f"R²   : {validation_r2:.4f}")

print("\nTest Performance")
print("-" * 40)

print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R²   : {test_r2:.4f}")

print("=" * 80)



# ============================================================
# Phase 19.7 : Feature Importance
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 19.7 : FEATURE IMPORTANCE")
print("=" * 80)

importance_df = pd.DataFrame({

    "Feature": feature_columns,

    "Importance": rf_model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)

print("=" * 80)



# ============================================================
# Phase 21.1 : Save Random Forest Model
# ============================================================

import joblib

print("\n")
print("=" * 80)
print("PHASE 21.1 : SAVE RANDOM FOREST MODEL")
print("=" * 80)

joblib.dump(
    rf_model,
    "random_forest_model.pkl"
)

print("Random Forest Model Saved Successfully")

print("=" * 80)