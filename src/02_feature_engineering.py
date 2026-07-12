import numpy as np 

import pandas as pd 


aqi_df = pd.read_csv(r"/Users/sohan/Documents/AI-Based-Air-Quality-Prediction-System/data/processed/preprocessed_aqi_dataset.csv")





# ============================================================
# Phase 16.1 : Time-Based Feature Engineering
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 16.1 : TIME-BASED FEATURE ENGINEERING")
print("=" * 80)

# ------------------------------------------------------------
# Ensure Datetime column is in datetime format
# ------------------------------------------------------------

aqi_df["Datetime"] = pd.to_datetime(aqi_df["Datetime"])

# ------------------------------------------------------------
# Extract Hour of the Day
# Range : 0 - 23
# ------------------------------------------------------------

aqi_df["Hour"] = aqi_df["Datetime"].dt.hour

# ------------------------------------------------------------
# Extract Day of the Month
# Range : 1 - 31
# ------------------------------------------------------------

aqi_df["Day"] = aqi_df["Datetime"].dt.day

# ------------------------------------------------------------
# Extract Month
# Range : 1 - 12
# ------------------------------------------------------------

aqi_df["Month"] = aqi_df["Datetime"].dt.month

# ------------------------------------------------------------
# Extract Year
# ------------------------------------------------------------

aqi_df["Year"] = aqi_df["Datetime"].dt.year

# ------------------------------------------------------------
# Extract Day of Week
# Monday = 0
# Sunday = 6
# ------------------------------------------------------------

aqi_df["DayOfWeek"] = aqi_df["Datetime"].dt.dayofweek

# ------------------------------------------------------------
# Weekend Indicator
# Saturday = 1
# Sunday = 1
# Weekdays = 0
# ------------------------------------------------------------

aqi_df["IsWeekend"] = (
    aqi_df["DayOfWeek"] >= 5
).astype(int)

# ------------------------------------------------------------
# Quarter of the Year
# Values : 1 - 4
# ------------------------------------------------------------

aqi_df["Quarter"] = aqi_df["Datetime"].dt.quarter

# ------------------------------------------------------------
# Season Feature
#
# Winter : Dec-Feb
# Summer : Mar-May
# Monsoon : Jun-Sep
# PostMonsoon : Oct-Nov
# ------------------------------------------------------------

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "PostMonsoon"


aqi_df["Season"] = aqi_df["Month"].apply(get_season)

# ------------------------------------------------------------
# Display newly created features
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("NEWLY CREATED TIME FEATURES")
print("=" * 80)

print(
    aqi_df[
        [
            "Datetime",
            "Hour",
            "Day",
            "Month",
            "Year",
            "DayOfWeek",
            "IsWeekend",
            "Quarter",
            "Season"
        ]
    ].head(10)
)

print("\n")
print("=" * 80)
print("TIME-BASED FEATURE ENGINEERING COMPLETED")
print("=" * 80)


# ------------------------------------------------------------
# Sort dataset before feature engineering
#
# Sorting is mandatory for:
# - Lag Features
# - Rolling Statistics
# - LSTM Sequence Generation
# ------------------------------------------------------------

aqi_df.sort_values(
    by=["StationId", "Datetime"],
    inplace=True
)

aqi_df.reset_index(
    drop=True,
    inplace=True
)

print("\nDataset sorted successfully.")

# ============================================================
# Phase 16.2 : Station-wise Lag Feature Engineering
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 16.2 : STATION-WISE LAG FEATURES")
print("=" * 80)

# ------------------------------------------------------------
# AQI Lag Features
# ------------------------------------------------------------

aqi_df["AQI_lag_1"] = (
    aqi_df.groupby("StationId")["AQI"].shift(1)
)

aqi_df["AQI_lag_3"] = (
    aqi_df.groupby("StationId")["AQI"].shift(3)
)

aqi_df["AQI_lag_6"] = (
    aqi_df.groupby("StationId")["AQI"].shift(6)
)

aqi_df["AQI_lag_12"] = (
    aqi_df.groupby("StationId")["AQI"].shift(12)
)

aqi_df["AQI_lag_24"] = (
    aqi_df.groupby("StationId")["AQI"].shift(24)
)

# ------------------------------------------------------------
# Pollutant Lag Features
# ------------------------------------------------------------

aqi_df["PM2.5_lag_1"] = (
    aqi_df.groupby("StationId")["PM2.5"].shift(1)
)

aqi_df["PM2.5_lag_3"] = (
    aqi_df.groupby("StationId")["PM2.5"].shift(3)
)

aqi_df["PM10_lag_1"] = (
    aqi_df.groupby("StationId")["PM10"].shift(1)
)

aqi_df["NO2_lag_1"] = (
    aqi_df.groupby("StationId")["NO2"].shift(1)
)

aqi_df["CO_lag_1"] = (
    aqi_df.groupby("StationId")["CO"].shift(1)
)

print("\nLag Features Created Successfully.")

print("\n")
print(aqi_df[
    [
        "StationId",
        "Datetime",
        "AQI",
        "AQI_lag_1",
        "AQI_lag_3",
        "AQI_lag_6",
        "AQI_lag_12",
        "AQI_lag_24"
    ]
].head(15))

print("\n")
print("=" * 80)
print("PHASE 16.2 COMPLETED")
print("=" * 80)


# ============================================================
# Phase 16.3 : Station-wise Rolling Window Features
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 16.3 : STATION-WISE ROLLING FEATURES")
print("=" * 80)

# ------------------------------------------------------------
# AQI Rolling Mean Features
# ------------------------------------------------------------

aqi_df["AQI_roll_mean_3"] = (
    aqi_df.groupby("StationId")["AQI"]
          .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

aqi_df["AQI_roll_mean_6"] = (
    aqi_df.groupby("StationId")["AQI"]
          .transform(lambda x: x.shift(1).rolling(window=6, min_periods=1).mean())
)

aqi_df["AQI_roll_mean_12"] = (
    aqi_df.groupby("StationId")["AQI"]
          .transform(lambda x: x.shift(1).rolling(window=12, min_periods=1).mean())
)

# ------------------------------------------------------------
# PM2.5 Rolling Mean Features
# ------------------------------------------------------------

aqi_df["PM2.5_roll_mean_3"] = (
    aqi_df.groupby("StationId")["PM2.5"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

aqi_df["PM2.5_roll_mean_6"] = (
    aqi_df.groupby("StationId")["PM2.5"]
          .transform(lambda x: x.rolling(window=6, min_periods=1).mean())
)

# ------------------------------------------------------------
# PM10 Rolling Mean Feature
# ------------------------------------------------------------

aqi_df["PM10_roll_mean_3"] = (
    aqi_df.groupby("StationId")["PM10"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

# ------------------------------------------------------------
# NO2 Rolling Mean Feature
# ------------------------------------------------------------

aqi_df["NO2_roll_mean_3"] = (
    aqi_df.groupby("StationId")["NO2"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

# ------------------------------------------------------------
# CO Rolling Mean Feature
# ------------------------------------------------------------

aqi_df["CO_roll_mean_3"] = (
    aqi_df.groupby("StationId")["CO"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

# ------------------------------------------------------------
# Preview Rolling Features
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("ROLLING FEATURE PREVIEW")
print("=" * 80)

print(
    aqi_df[
        [
            "StationId",
            "Datetime",
            "AQI",
            "AQI_roll_mean_3",
            "AQI_roll_mean_6",
            "AQI_roll_mean_12",
            "PM2.5_roll_mean_3",
            "PM2.5_roll_mean_6"
        ]
    ].head(15)
)

print("\n")
print("=" * 80)
print("PHASE 16.3 COMPLETED")
print("=" * 80)


# ============================================================
# Phase 16.4 : Final Feature Validation
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 16.4 : FINAL FEATURE VALIDATION")
print("=" * 80)

# ------------------------------------------------------------
# Display Dataset Shape
# ------------------------------------------------------------

print(f"\nDataset Shape : {aqi_df.shape}")

# ------------------------------------------------------------
# Display Column Names
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("FINAL FEATURE LIST")
print("=" * 80)

print(aqi_df.columns.tolist())

# ------------------------------------------------------------
# Check Missing Values
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("MISSING VALUE SUMMARY")
print("=" * 80)

missing = aqi_df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

print(missing)

# ------------------------------------------------------------
# Check Duplicate Rows
# ------------------------------------------------------------

duplicates = aqi_df.duplicated().sum()

print("\n")
print("=" * 80)
print("DUPLICATE ROWS")
print("=" * 80)

print(f"Duplicate Rows : {duplicates}")

# ------------------------------------------------------------
# Display Data Types
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("DATA TYPES")
print("=" * 80)

print(aqi_df.dtypes)

# ------------------------------------------------------------
# Preview Final Dataset
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("FINAL DATASET PREVIEW")
print("=" * 80)

print(aqi_df.head())

# ------------------------------------------------------------
# Save Feature Engineered Dataset
# ------------------------------------------------------------

aqi_df.to_csv(
    "feature_engineered_dataset.csv",
    index=False
)

print("\n")
print("=" * 80)
print("FEATURE ENGINEERED DATASET SAVED SUCCESSFULLY")
print("File : feature_engineered_dataset.csv")
print("=" * 80)



feature_columns = [

    # --------------------------------------------------------
    # Current Pollutant Concentrations
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Time Features
    # --------------------------------------------------------

    "Hour",
    "Day",
    "Month",
    "Year",
    "DayOfWeek",
    "IsWeekend",
    "Quarter",

    # --------------------------------------------------------
    # AQI Lag Features
    # --------------------------------------------------------

    "AQI_lag_1",
    "AQI_lag_3",
    "AQI_lag_6",
    "AQI_lag_12",
    "AQI_lag_24",

    # --------------------------------------------------------
    # Pollutant Lag Features
    # --------------------------------------------------------

    "PM2.5_lag_1",
    "PM2.5_lag_3",
    "PM10_lag_1",
    "NO2_lag_1",
    "CO_lag_1",

    # --------------------------------------------------------
    # Rolling Mean Features
    # --------------------------------------------------------

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
# Phase 17.1 : Model Feature Missing Value Analysis
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 17.1 : FEATURE-WISE MISSING VALUE ANALYSIS")
print("=" * 80)

# ------------------------------------------------------------
# Check missing values only in the selected model features
# ------------------------------------------------------------

feature_missing = (
    aqi_df[feature_columns]
    .isnull()
    .sum()
    .sort_values(ascending=False)
)

feature_missing = feature_missing[feature_missing > 0]

print(feature_missing)

print("\n")
print("=" * 80)
print("MISSING PERCENTAGE")
print("=" * 80)

feature_missing_percentage = (
    feature_missing / len(aqi_df) * 100
).round(2)

print(feature_missing_percentage)


# ============================================================
# Phase 17.2 : Feature Impact Analysis
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 17.2 : FEATURE IMPACT ANALYSIS")
print("=" * 80)

baseline_rows = len(
    aqi_df[feature_columns + [target_column]].dropna()
)

print(f"\nCurrent Training Rows : {baseline_rows:,}\n")

results = []

for feature in feature_columns:

    temp_features = [
        col for col in feature_columns
        if col != feature
    ]

    remaining_rows = len(
        aqi_df[temp_features + [target_column]].dropna()
    )

    recovered = remaining_rows - baseline_rows

    results.append([
        feature,
        remaining_rows,
        recovered
    ])

impact_df = pd.DataFrame(
    results,
    columns=[
        "Feature Removed",
        "Training Rows",
        "Rows Recovered"
    ]
)

impact_df = impact_df.sort_values(
    "Rows Recovered",
    ascending=False
)

print(impact_df)


















model_df = aqi_df[
    feature_columns + [target_column]
].dropna().copy()


print("=" * 80)
print("MODEL DATASET SUMMARY")
print("=" * 80)

print(f"Original Dataset : {len(aqi_df):,}")
print(f"Training Dataset : {len(model_df):,}")
print(f"Rows Removed     : {len(aqi_df)-len(model_df):,}")




print(model_df.head())