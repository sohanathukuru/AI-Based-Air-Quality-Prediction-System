# ============================================================
# AI-Based Air Quality Prediction System
#
# Phase 20 : LSTM Model
#
# Author : Athukuru Sohan
# ============================================================

import pandas as pd
import numpy as np

print("\n")
print("=" * 80)
print("PHASE 20.1 : LSTM MODEL INITIALIZATION")
print("=" * 80)


from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)


# ============================================================
# Load Feature Engineered Dataset
# ============================================================

aqi_df = pd.read_csv("/Users/sohan/Documents/AI-Based-Air-Quality-Prediction-System/data/processed/feature_engineered_dataset.csv")

aqi_df["Datetime"] = pd.to_datetime(aqi_df["Datetime"])

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
# Phase 20.2 : Feature Scaling
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 20.2 : FEATURE SCALING")
print("=" * 80)

# ------------------------------------------------------------
# Initialize StandardScaler
# ------------------------------------------------------------

scaler = StandardScaler()

# ------------------------------------------------------------
# Fit ONLY on Training Data
# ------------------------------------------------------------

X_train_scaled = scaler.fit_transform(X_train)

# ------------------------------------------------------------
# Transform Validation and Test Data
# ------------------------------------------------------------

X_validation_scaled = scaler.transform(X_validation)

X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed Successfully")

print()

print("Training Shape   :", X_train_scaled.shape)
print("Validation Shape :", X_validation_scaled.shape)
print("Testing Shape    :", X_test_scaled.shape)

print("=" * 80)


# ============================================================
# Phase 20.3 : LSTM Sequence Generation
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 20.3 : LSTM SEQUENCE GENERATION")
print("=" * 80)

# ------------------------------------------------------------
# Number of Previous Hours Used
# ------------------------------------------------------------

TIME_STEPS = 24

# ------------------------------------------------------------
# Function to Create Sequences
# ------------------------------------------------------------

def create_sequences(features, target, time_steps):

    X_sequence = []
    y_sequence = []

    for i in range(len(features) - time_steps):

        X_sequence.append(
            features[i : i + time_steps]
        )

        y_sequence.append(
            target.iloc[i + time_steps]
        )

    return np.array(X_sequence), np.array(y_sequence)


# ------------------------------------------------------------
# Training Sequences
# ------------------------------------------------------------

X_train_sequences = []
y_train_sequences = []

for station_id, station_df in train_df.groupby("StationId"):

    station_df = station_df.sort_values("Datetime")

    X_station = scaler.transform(
        station_df[feature_columns]
    )

    y_station = station_df[target_column]

    X_seq, y_seq = create_sequences(
        X_station,
        y_station,
        TIME_STEPS
    )

    X_train_sequences.append(X_seq)
    y_train_sequences.append(y_seq)

X_train_sequences = np.concatenate(X_train_sequences)
y_train_sequences = np.concatenate(y_train_sequences)



# ------------------------------------------------------------
# Validation Sequences
# ------------------------------------------------------------

X_validation_sequences = []
y_validation_sequences = []

for station_id, station_df in validation_df.groupby("StationId"):

    station_df = station_df.sort_values("Datetime")

    X_station = scaler.transform(
        station_df[feature_columns]
    )

    y_station = station_df[target_column]

    X_seq, y_seq = create_sequences(
        X_station,
        y_station,
        TIME_STEPS
    )

    X_validation_sequences.append(X_seq)
    y_validation_sequences.append(y_seq)

X_validation_sequences = np.concatenate(X_validation_sequences)
y_validation_sequences = np.concatenate(y_validation_sequences)


# ------------------------------------------------------------
# Testing Sequences
# ------------------------------------------------------------

X_test_sequences = []
y_test_sequences = []

for station_id, station_df in test_df.groupby("StationId"):

    station_df = station_df.sort_values("Datetime")

    X_station = scaler.transform(
        station_df[feature_columns]
    )

    y_station = station_df[target_column]

    X_seq, y_seq = create_sequences(
        X_station,
        y_station,
        TIME_STEPS
    )

    X_test_sequences.append(X_seq)
    y_test_sequences.append(y_seq)

X_test_sequences = np.concatenate(X_test_sequences)
y_test_sequences = np.concatenate(y_test_sequences)



print("\n")
print("=" * 80)
print("LSTM DATASET SHAPES")
print("=" * 80)

print("X_train :", X_train_sequences.shape)
print("y_train :", y_train_sequences.shape)

print()

print("X_validation :", X_validation_sequences.shape)
print("y_validation :", y_validation_sequences.shape)

print()

print("X_test :", X_test_sequences.shape)
print("y_test :", y_test_sequences.shape)

print("=" * 80)


# ============================================================
# Phase 20.4 : Build LSTM Model
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 20.4 : BUILD LSTM MODEL")
print("=" * 80)

# ------------------------------------------------------------
# Build Sequential Model
# ------------------------------------------------------------

lstm_model = Sequential()

# ------------------------------------------------------------
# First LSTM Layer
# ------------------------------------------------------------

lstm_model.add(

    LSTM(

        units=64,

        return_sequences=True,

        input_shape=(
            TIME_STEPS,
            len(feature_columns)
        )

    )

)

# ------------------------------------------------------------
# Dropout Layer
# ------------------------------------------------------------

lstm_model.add(

    Dropout(0.20)

)

# ------------------------------------------------------------
# Second LSTM Layer
# ------------------------------------------------------------

lstm_model.add(

    LSTM(

        units=32,

        return_sequences=False

    )

)

# ------------------------------------------------------------
# Dropout Layer
# ------------------------------------------------------------

lstm_model.add(

    Dropout(0.20)

)

# ------------------------------------------------------------
# Dense Hidden Layer
# ------------------------------------------------------------

lstm_model.add(

    Dense(

        units=16,

        activation="relu"

    )

)

# ------------------------------------------------------------
# Output Layer
# ------------------------------------------------------------

lstm_model.add(

    Dense(

        units=1

    )

)

print("LSTM Model Built Successfully")

print("=" * 80)



# ============================================================
# Phase 20.5 : Compile LSTM Model
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 20.5 : COMPILE LSTM MODEL")
print("=" * 80)

lstm_model.compile(

    optimizer="adam",

    loss="mse",

    metrics=["mae"]

)

print("LSTM Model Compiled Successfully")

print("=" * 80)

print("\n")

lstm_model.summary()


# ============================================================
# Phase 20.6 : Train LSTM Model
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 20.6 : LSTM TRAINING")
print("=" * 80)

# ------------------------------------------------------------
# Early Stopping
# ------------------------------------------------------------

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

# ------------------------------------------------------------
# Train LSTM
# ------------------------------------------------------------

history = lstm_model.fit(

    X_train_sequences,

    y_train_sequences,

    validation_data=(

        X_validation_sequences,

        y_validation_sequences

    ),

    epochs=20,

    batch_size=64,

    callbacks=[early_stopping],

    verbose=1

)

print("\nLSTM Training Completed Successfully")

print("=" * 80)


# ============================================================
# Phase 20.7 : Validation Prediction
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\n")
print("=" * 80)
print("PHASE 20.7 : VALIDATION PREDICTION")
print("=" * 80)

# ------------------------------------------------------------
# Validation Prediction
# ------------------------------------------------------------

y_validation_pred = lstm_model.predict(
    X_validation_sequences,
    verbose=1
)

# Convert to 1D array
y_validation_pred = y_validation_pred.flatten()

print("\nValidation Prediction Completed Successfully")

print(f"Validation Predictions : {len(y_validation_pred):,}")

print("=" * 80)


# ============================================================
# Phase 20.8 : Test Prediction
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 20.8 : TEST PREDICTION")
print("=" * 80)

# ------------------------------------------------------------
# Test Prediction
# ------------------------------------------------------------

y_test_pred = lstm_model.predict(
    X_test_sequences,
    verbose=1
)

# Convert to 1D array
y_test_pred = y_test_pred.flatten()

print("\nTest Prediction Completed Successfully")

print(f"Test Predictions : {len(y_test_pred):,}")

print("=" * 80)


# ============================================================
# Phase 20.9 : LSTM Evaluation
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 20.9 : LSTM EVALUATION")
print("=" * 80)

# ------------------------------------------------------------
# Validation Metrics
# ------------------------------------------------------------

validation_mae = mean_absolute_error(
    y_validation_sequences,
    y_validation_pred
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation_sequences,
        y_validation_pred
    )
)

validation_r2 = r2_score(
    y_validation_sequences,
    y_validation_pred
)

# ------------------------------------------------------------
# Test Metrics
# ------------------------------------------------------------

test_mae = mean_absolute_error(
    y_test_sequences,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test_sequences,
        y_test_pred
    )
)

test_r2 = r2_score(
    y_test_sequences,
    y_test_pred
)

# ------------------------------------------------------------
# Display Results
# ------------------------------------------------------------

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
# Phase 21.2 : Save LSTM Model
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 21.2 : SAVE LSTM MODEL")
print("=" * 80)

lstm_model.save(
    "lstm_model.keras"
)

print("LSTM Model Saved Successfully")

print("=" * 80)