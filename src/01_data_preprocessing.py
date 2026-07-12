import pandas as pd
import numpy as np

# Load the final filtered dataset
aqi_df = pd.read_csv("/Users/sohan/Documents/AI-Based-Air-Quality-Prediction-System/data/raw/Delhi_Hyderabad_Station_Hour.csv")

# ============================================================
# Step 1: Convert the 'Datetime' Column to Datetime Format
# ============================================================

aqi_df["Datetime"] = pd.to_datetime(aqi_df["Datetime"])

print(aqi_df["Datetime"].dtype)

print(aqi_df["Datetime"].head())

aqi_df = aqi_df.sort_values(
    by=["City", "StationId", "Datetime"]
).reset_index(drop=True)

print(aqi_df.head())

print(aqi_df.tail())

# ============================================================
# Step 3: Analyze Missing Values
# ============================================================
#
# Why are we doing this?
# ----------------------
# Real-world air quality datasets often contain missing values
# because:
# • Sensors may temporarily stop recording.
# • Equipment may undergo maintenance.
# • Communication failures may occur.
#
# Before handling missing values, we must first determine:
# • Which columns contain missing values.
# • How many values are missing.
# • The percentage of missing values in each column.
#
# This analysis helps us decide whether to:
# • Impute missing values.
# • Drop unnecessary columns.
# • Remove rows (if required).
# ============================================================

# Count missing values in each column
missing_values = aqi_df.isnull().sum()

# Calculate percentage of missing values
missing_percentage = (aqi_df.isnull().sum() / len(aqi_df)) * 100

# Create a summary table
missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage (%)": missing_percentage.round(2)
})

# Sort columns from highest to lowest missing percentage
missing_summary = missing_summary.sort_values(
    by="Missing Percentage (%)",
    ascending=False
)

# Display the missing value summary
print("\n========== Missing Value Summary ==========\n")
print(missing_summary)

# ============================================================
# Step 4: Remove Columns with Excessive Missing Values
# ============================================================
#
# Why are we doing this?
# ----------------------
# The 'Xylene' column contains approximately 77% missing values.
# Such a high percentage of missing data makes reliable
# imputation difficult and may introduce noise into the model.
#
# Since Xylene contributes relatively less to AQI prediction
# compared to major pollutants like PM2.5, PM10, NO2, CO,
# and O3, removing this column helps improve data quality
# while simplifying preprocessing.
# ============================================================

# Remove the Xylene column
aqi_df.drop(columns=["Xylene"], inplace=True)

# Verify that the column has been removed
print("Remaining Columns:\n")
print(aqi_df.columns)

# ============================================================
# Step 5A: Visualize Missing Values by Column
# ============================================================
#
# Why are we doing this?
# ----------------------
# This chart clearly shows the percentage of missing values
# present in each feature.
#
# It helps identify:
# • Columns with excessive missing data.
# • Columns suitable for imputation.
# • Columns that may need to be removed.
# ============================================================

# import matplotlib.pyplot as plt

# # Calculate missing percentage
# missing_percentage = (aqi_df.isnull().sum() / len(aqi_df)) * 100

# # Sort values
# missing_percentage = missing_percentage.sort_values(ascending=False)

# # Create figure
# plt.figure(figsize=(12,6))

# # Plot bar chart
# plt.bar(missing_percentage.index, missing_percentage.values)

# # Labels
# plt.title("Missing Value Percentage by Feature")
# plt.xlabel("Features")
# plt.ylabel("Missing Percentage (%)")

# plt.xticks(rotation=60)

# plt.grid(axis="y", linestyle="--", alpha=0.5)

# plt.show()

# # ============================================================
# # Step 5B: Missing AQI Values per Monitoring Station
# # ============================================================

# station_missing = (
#     aqi_df.groupby("StationId")["AQI"]
#     .apply(lambda x: x.isnull().sum())
#     .sort_values(ascending=False)
# )

# plt.figure(figsize=(15,5))

# plt.bar(station_missing.index, station_missing.values)

# plt.title("Missing AQI Values by Station")

# plt.xlabel("Station ID")

# plt.ylabel("Missing AQI Count")

# plt.xticks(rotation=90)

# plt.show()

# # ============================================================
# # Step 5C: Missing AQI Values Over Time
# # ============================================================

# aqi_missing = aqi_df.copy()

# aqi_missing["AQI_Missing"] = aqi_missing["AQI"].isnull()

# missing_per_day = (
#     aqi_missing.groupby(aqi_missing["Datetime"].dt.date)["AQI_Missing"]
#     .sum()
# )

# plt.figure(figsize=(15,5))

# plt.plot(missing_per_day.index,
#          missing_per_day.values)

# plt.title("Daily Missing AQI Values")

# plt.xlabel("Date")

# plt.ylabel("Number of Missing AQI Values")

# plt.grid(True)

# plt.show()

# ============================================================
# Step 5D: Missing Data Matrix
# ============================================================

# import missingno as msno
# import matplotlib.pyplot as plt

# plt.figure(figsize=(15,8))

# msno.matrix(aqi_df)

# plt.show()

# ============================================================
# Step 6: Strict Station-wise Time Interpolation
# ============================================================
#
# Why are we using this approach?
# -------------------------------
# This implementation performs interpolation separately for
# each monitoring station and each pollutant.
#
# Missing values are interpolated ONLY when the elapsed time
# between the previous valid observation and the next valid
# observation is less than or equal to six hours.
#
# Longer outages are intentionally left as missing values
# because linear interpolation over long periods may generate
# unrealistic pollutant concentrations.
#
# AQI is NOT interpolated because it is the prediction target.
# ============================================================



# ------------------------------------------------------------
# Pollutant columns to interpolate
# ------------------------------------------------------------

pollutant_columns = [
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
    "Toluene"
]

# Maximum allowed interpolation gap
MAX_GAP = pd.Timedelta(hours=6)

# ------------------------------------------------------------
# Process each monitoring station independently
# ------------------------------------------------------------

for station in aqi_df["StationId"].unique():

    station_mask = aqi_df["StationId"] == station

    station_df = (
        aqi_df.loc[station_mask]
        .sort_values("Datetime")
        .copy()
    )

  

# ============================================================
# Phase 3 : Linear Interpolation Function
# ============================================================
#
# Purpose
# -------
# This function performs linear interpolation for a single
# missing-value block.
#
# It does NOT know anything about stations or AQI datasets.
#
# It simply receives:
#
# • Previous valid value
# • Next valid value
# • Previous timestamp
# • Next timestamp
# • Missing timestamps
#
# and returns the interpolated values.
#
# This modular design makes the function reusable throughout
# the project.
# ============================================================

def interpolate_block(
    previous_value,
    next_value,
    previous_time,
    next_time,
    missing_times
):
    """
    Linearly interpolate values for a missing block.

    Parameters
    ----------
    previous_value : float
        Value before the missing block.

    next_value : float
        Value after the missing block.

    previous_time : pandas.Timestamp
        Timestamp before the missing block.

    next_time : pandas.Timestamp
        Timestamp after the missing block.

    missing_times : list[pandas.Timestamp]
        List of timestamps corresponding to the missing values.

    Returns
    -------
    list[float]
        Interpolated values.
    """

    interpolated_values = []

    # Total elapsed time between valid observations (seconds)
    total_seconds = (
        next_time - previous_time
    ).total_seconds()

    for timestamp in missing_times:

        # Time elapsed from previous valid observation
        elapsed_seconds = (
            timestamp - previous_time
        ).total_seconds()

        # Linear interpolation ratio
        ratio = elapsed_seconds / total_seconds

        # Linear interpolation equation
        interpolated_value = (
            previous_value
            + (next_value - previous_value) * ratio
        )

        interpolated_values.append(interpolated_value)

    return interpolated_values





# ============================================================
# Phase 5 : Generic Station-wise Interpolation Engine
# ============================================================
#
# Purpose
# -------
# Interpolate ONE pollutant column for ONE monitoring station.
#
# This function is reusable for every pollutant parameter.
#
# Workflow
# --------
# 1. Detect consecutive missing blocks.
# 2. Compute elapsed time between surrounding observations.
# 3. Interpolate ONLY if elapsed time <= MAX_GAP_HOURS.
# 4. Leave longer gaps unchanged.
# ============================================================

MAX_GAP_HOURS = 6


def interpolate_station(station_df, column_name):

    # --------------------------------------------------------
    # Work on a copy of the station data
    # --------------------------------------------------------

    station_df = station_df.copy()

    # Sort chronologically
    station_df = (
        station_df
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

    # Select pollutant column
    series = station_df[column_name]

    i = 0

    while i < len(series):

        # Skip valid observations
        if not pd.isna(series.iloc[i]):
            i += 1
            continue

        # ----------------------------------------------------
        # Beginning of missing block
        # ----------------------------------------------------

        start = i

        while i < len(series) and pd.isna(series.iloc[i]):
            i += 1

        end = i - 1

        # Missing block at beginning
        if start == 0:
            continue

        # Missing block at end
        if end == len(series) - 1:
            continue

        previous_time = station_df.loc[start - 1, "Datetime"]
        next_time = station_df.loc[end + 1, "Datetime"]

        elapsed_hours = (
            next_time - previous_time
        ).total_seconds() / 3600

        # Skip long gaps
        if elapsed_hours > MAX_GAP_HOURS:
            continue

        previous_value = series.iloc[start - 1]
        next_value = series.iloc[end + 1]

        missing_times = list(
            station_df.loc[start:end, "Datetime"]
        )

        interpolated_values = interpolate_block(

            previous_value,
            next_value,
            previous_time,
            next_time,
            missing_times

        )

        # ----------------------------------------------------
        # Write interpolated values
        # ----------------------------------------------------

        for row_index, value in zip(
            range(start, end + 1),
            interpolated_values
        ):

            series.iloc[row_index] = value

    station_df[column_name] = series

    return station_df



# ============================================================
# Step 6 : Apply Station-wise Time Interpolation
# ============================================================
#
# Purpose
# -------
# This step applies the validated interpolation engine to the
# real Delhi-Hyderabad air quality dataset.
#
# Strategy
# --------
# • Interpolate pollutant values separately for every station.
# • Interpolate ONLY when the elapsed time between two valid
#   observations is less than or equal to 6 hours.
# • Longer gaps are intentionally left as missing values.
# • AQI is NOT interpolated because it is the prediction target.
#
# Benefits
# --------
# • Preserves the temporal behaviour of each station.
# • Prevents unrealistic interpolation across long outages.
# • Produces higher-quality data for machine learning.
# ============================================================

# ------------------------------------------------------------
# List of pollutant columns to be interpolated
# ------------------------------------------------------------

pollutant_columns = [

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
    "Toluene"

]

print("\n")
print("=" * 80)
print("STARTING STATION-WISE INTERPOLATION")
print("=" * 80)

# ------------------------------------------------------------
# Apply interpolation one pollutant at a time
# ------------------------------------------------------------

for pollutant in pollutant_columns:

    print("\n")
    print("-" * 70)
    print(f"Interpolating : {pollutant}")

    # Count missing values before interpolation
    missing_before = aqi_df[pollutant].isna().sum()

    # --------------------------------------------------------
    # Apply interpolation separately for every station
    # --------------------------------------------------------

    interpolated_groups = []

    for station_id, station_data in aqi_df.groupby("StationId"):

        station_result = interpolate_station(
            station_data,
            pollutant
        )

        interpolated_groups.append(station_result)

    # --------------------------------------------------------
    # Combine all stations back into one dataframe
    # --------------------------------------------------------

    aqi_df = (
        pd.concat(interpolated_groups)
        .sort_index()
        .reset_index(drop=True)
    )

    # Count missing values after interpolation
    missing_after = aqi_df[pollutant].isna().sum()

    recovered = missing_before - missing_after

    print(f"Missing Before : {missing_before:,}")
    print(f"Missing After  : {missing_after:,}")
    print(f"Recovered      : {recovered:,}")

print("\n")
print("=" * 80)
print("STATION-WISE INTERPOLATION COMPLETED SUCCESSFULLY")
print("=" * 80)


# ============================================================
# Step 7 : Missing Value Summary After Interpolation
# ============================================================

print("\n")
print("=" * 80)
print("MISSING VALUE SUMMARY AFTER INTERPOLATION")
print("=" * 80)

missing_summary = pd.DataFrame({

    "Missing Values": aqi_df.isnull().sum(),

    "Missing Percentage (%)":
        (
            aqi_df.isnull().mean() * 100
        ).round(2)

})

print(
    missing_summary.sort_values(
        by="Missing Values",
        ascending=False
    )
)


aqi_df.drop(columns=['AQI_Bucket'] , inplace = True)

print(aqi_df.columns)


# ============================================================
# Step 10.1 : Analyze Missing AQI Values
# ============================================================
#
# Purpose
# -------
# AQI is the prediction target for this supervised learning
# project. Before removing rows with missing AQI values,
# determine how many such rows exist and what percentage of
# the dataset they represent.
#
# This analysis helps us decide whether removing these rows
# is appropriate or whether an alternative strategy is needed.
# ============================================================

missing_aqi = aqi_df["AQI"].isna().sum()
total_rows = len(aqi_df)

missing_percentage = (missing_aqi / total_rows) * 100

print("\n")
print("=" * 80)
print("AQI MISSING VALUE ANALYSIS")
print("=" * 80)

print(f"Total Rows              : {total_rows:,}")
print(f"Rows with Missing AQI   : {missing_aqi:,}")
print(f"Missing Percentage      : {missing_percentage:.2f}%")

print("=" * 80)

# ============================================================
# Step 10.2 : Check for Duplicate Records
# ============================================================

duplicate_rows = aqi_df.duplicated().sum()

print("\n")
print("=" * 80)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 80)

print(f"Duplicate Rows : {duplicate_rows:,}")

print("=" * 80)

# ============================================================
# Step 10.3 : Check for Invalid Sensor Readings
# ============================================================

pollutant_columns = [
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
    "Toluene"
]

print("\n")
print("=" * 80)
print("INVALID SENSOR READING ANALYSIS")
print("=" * 80)

for column in pollutant_columns:

    invalid_count = (aqi_df[column] < 0).sum()

    print(f"{column:<10} : {invalid_count:,}")

print("=" * 80)


# ============================================================
# Step 10.4 : Validate AQI Range
# ============================================================

invalid_aqi = (
    (aqi_df["AQI"] < 0) |
    (aqi_df["AQI"] > 500)
).sum()

print("\n")
print("=" * 80)
print("AQI RANGE VALIDATION")
print("=" * 80)

print(f"Invalid AQI Values : {invalid_aqi:,}")

print("=" * 80)



# ============================================================
# Step 10.5 : Inspect AQI Values Outside CPCB Range
# ============================================================

invalid_aqi_rows = aqi_df[
    (aqi_df["AQI"].notna()) &
    (
        (aqi_df["AQI"] < 0) |
        (aqi_df["AQI"] > 500)
    )
]

print("\n")
print("=" * 80)
print("INVALID AQI SAMPLE")
print("=" * 80)

print(invalid_aqi_rows[["StationId", "Datetime", "AQI"]].head(20))

print("\nUnique Invalid AQI Values:")

print(sorted(invalid_aqi_rows["AQI"].unique()))

print("=" * 80)




invalid_aqi = aqi_df["AQI"] < 0

print("=" * 80)
print("AQI VALIDATION")
print("=" * 80)

print("Negative AQI Values :", invalid_aqi.sum())

print("=" * 80)



# ============================================================
# Step 10.6 : Station-wise Missing AQI Analysis
# ============================================================
#
# Purpose
# -------
# Before removing rows with missing AQI values, analyze how
# they are distributed across monitoring stations.
#
# If missing AQI values are concentrated in only a few stations,
# blindly removing them may introduce station bias.
#
# This analysis helps determine whether removing rows with
# missing AQI is an appropriate strategy.
# ============================================================

station_summary = (
    aqi_df
    .groupby("StationId")
    .agg(
        Total_Rows=("AQI", "size"),
        Missing_AQI=("AQI", lambda x: x.isna().sum())
    )
)

station_summary["Missing_Percentage"] = (
    station_summary["Missing_AQI"] /
    station_summary["Total_Rows"] * 100
)

station_summary = (
    station_summary
    .sort_values("Missing_Percentage", ascending=False)
)

print("\n")
print("=" * 80)
print("STATION-WISE MISSING AQI ANALYSIS")
print("=" * 80)

print(station_summary)

print("=" * 80)


print("\n")
print("=" * 80)
print("TOP 15 STATIONS WITH HIGHEST MISSING AQI")
print("=" * 80)

print(station_summary.head(15))

print("=" * 80)


print("\n")
print("=" * 80)
print("STATIONS WITH 100% MISSING AQI")
print("=" * 80)

fully_missing = station_summary[
    station_summary["Missing_Percentage"] == 100
]

if len(fully_missing) == 0:
    print("None")
else:
    print(fully_missing)

print("=" * 80)



# ============================================================
# Step 11 : Remove Unusable Training Samples
# ============================================================
#
# Purpose
# -------
# This project uses supervised regression where AQI is the
# prediction target.
#
# Samples without an AQI value cannot be used for training.
#
# Before removing rows with missing AQI, remove monitoring
# stations that contain no AQI observations at all.
#
# Station DL011 contains 100% missing AQI values and therefore
# contributes no usable training samples.
# ============================================================

print("\n")
print("=" * 80)
print("REMOVING UNUSABLE TRAINING SAMPLES")
print("=" * 80)

# ------------------------------------------------------------
# Store initial row count
# ------------------------------------------------------------

initial_rows = len(aqi_df)

# ------------------------------------------------------------
# Remove stations with 100% missing AQI
# ------------------------------------------------------------

stations_to_remove = ["DL011"]

rows_before_station_removal = len(aqi_df)

aqi_df = aqi_df[
    ~aqi_df["StationId"].isin(stations_to_remove)
].copy()

rows_after_station_removal = len(aqi_df)

removed_station_rows = (
    rows_before_station_removal -
    rows_after_station_removal
)

print(f"Stations Removed                  : {stations_to_remove}")
print(f"Rows Removed (DL011)              : {removed_station_rows:,}")

# ------------------------------------------------------------
# Remove rows with missing AQI
# ------------------------------------------------------------

rows_before_dropna = len(aqi_df)

aqi_df.dropna(subset=["AQI"], inplace=True)

rows_after_dropna = len(aqi_df)

removed_missing_aqi = (
    rows_before_dropna -
    rows_after_dropna
)

# ------------------------------------------------------------
# Final Summary
# ------------------------------------------------------------

total_removed = initial_rows - len(aqi_df)

retained_percentage = (
    len(aqi_df) / initial_rows
) * 100

print("\n")
print("=" * 80)
print("DATASET CLEANING SUMMARY")
print("=" * 80)

print(f"Initial Dataset Size              : {initial_rows:,}")
print(f"Rows Removed (DL011)              : {removed_station_rows:,}")
print(f"Rows Removed (Missing AQI)        : {removed_missing_aqi:,}")
print(f"Total Rows Removed                : {total_removed:,}")
print(f"Final Dataset Size                : {len(aqi_df):,}")
print(f"Data Retained                     : {retained_percentage:.2f}%")

print("=" * 80)



# ============================================================
# Phase 15.1 : Missing Value Reassessment
# ============================================================
#
# Purpose
# -------
# Re-evaluate the missing data pattern after completing the
# initial preprocessing pipeline.
#
# Changes already performed:
#   • Xylene removed
#   • Station-wise interpolation (≤ 6-hour gaps)
#   • AQI_Bucket removed
#   • Station DL011 removed
#   • Rows with missing AQI removed
#
# This visualization helps identify the remaining missing
# pollutant values, which primarily correspond to long sensor
# outages (> 6 hours), beginning gaps, and ending gaps.
# ============================================================

import missingno as msno
import matplotlib.pyplot as plt

print("\n")
print("=" * 80)
print("PHASE 15.1 : MISSING VALUE REASSESSMENT")
print("=" * 80)

plt.figure(figsize=(18, 8))

msno.matrix(
    aqi_df,
    figsize=(18, 8),
    sparkline=True,
    fontsize=10
)

plt.title(
    "Missing Value Matrix After Data Cleaning",
    fontsize=16,
    pad=20
)

plt.tight_layout()
plt.show()

print("=" * 80)
print("Missing value matrix generated successfully.")
print("=" * 80)


# ============================================================
# Phase 15.3 : Long Gap Analysis
# ============================================================
#
# Purpose
# -------
# Analyse every remaining missing block after interpolation.
#
# This helps determine which imputation strategy should be
# applied next.
#
# Categories
# ----------
# <= 6 Hours
# 7 - 24 Hours
# 1 - 3 Days
# > 3 Days
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 15.3 : LONG GAP ANALYSIS")
print("=" * 80)

pollutant_columns = [
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
    "Toluene"
]

gap_summary = []

for pollutant in pollutant_columns:

    less_equal_6 = 0
    between_7_24 = 0
    between_1_3_days = 0
    greater_3_days = 0

    for station in aqi_df["StationId"].unique():

        station_df = (
            aqi_df[aqi_df["StationId"] == station]
            .sort_values("Datetime")
            .reset_index(drop=True)
        )

        is_missing = station_df[pollutant].isna()

        i = 0

        while i < len(station_df):

            if not is_missing.iloc[i]:
                i += 1
                continue

            start = i

            while i < len(station_df) and is_missing.iloc[i]:
                i += 1

            end = i - 1

            # Ignore beginning gaps
            if start == 0:
                continue

            # Ignore ending gaps
            if end == len(station_df) - 1:
                continue

            prev_time = station_df.loc[start - 1, "Datetime"]
            next_time = station_df.loc[end + 1, "Datetime"]

            elapsed_hours = (
                next_time - prev_time
            ).total_seconds() / 3600

            if elapsed_hours <= 6:
                less_equal_6 += 1

            elif elapsed_hours <= 24:
                between_7_24 += 1

            elif elapsed_hours <= 72:
                between_1_3_days += 1

            else:
                greater_3_days += 1

    gap_summary.append([
        pollutant,
        less_equal_6,
        between_7_24,
        between_1_3_days,
        greater_3_days
    ])

gap_df = pd.DataFrame(
    gap_summary,
    columns=[
        "Pollutant",
        "<=6 Hours",
        "7-24 Hours",
        "1-3 Days",
        ">3 Days"
    ]
)

print("\n")
print("=" * 80)
print("LONG GAP DISTRIBUTION")
print("=" * 80)
print(gap_df.to_string(index=False))

print("\n")
print("=" * 80)
print("TOTAL GAPS")
print("=" * 80)

print(gap_df.sum(numeric_only=True))

print("=" * 80)



# ============================================================
# Phase 15.3.1 : Detailed Missing Gap Analysis
# ============================================================
#
# Purpose
# -------
# Analyse every remaining missing block after interpolation.
#
# For every pollutant, determine:
#
# • Number of missing blocks
# • Average gap length
# • Median gap length
# • Longest gap
# • Station having the longest gap
# • Start time of longest gap
# • End time of longest gap
#
# This analysis determines whether Kalman filtering is
# appropriate before implementing it.
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 15.3.1 : DETAILED GAP ANALYSIS")
print("=" * 80)

pollutant_columns = [
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
    "Toluene"
]

summary = []

for pollutant in pollutant_columns:

    gap_lengths = []

    longest_gap = 0
    longest_station = None
    longest_start = None
    longest_end = None

    # -----------------------------------------------
    # Analyse station by station
    # -----------------------------------------------

    for station in aqi_df["StationId"].unique():

        station_df = (
            aqi_df[aqi_df["StationId"] == station]
            .sort_values("Datetime")
            .reset_index(drop=True)
        )

        is_missing = station_df[pollutant].isna()

        i = 0

        while i < len(station_df):

            if not is_missing.iloc[i]:
                i += 1
                continue

            start = i

            while i < len(station_df) and is_missing.iloc[i]:
                i += 1

            end = i - 1

            # Ignore beginning gaps
            if start == 0:
                continue

            # Ignore ending gaps
            if end == len(station_df) - 1:
                continue

            gap_size = end - start + 1

            gap_lengths.append(gap_size)

            if gap_size > longest_gap:

                longest_gap = gap_size

                longest_station = station

                longest_start = station_df.loc[start, "Datetime"]

                longest_end = station_df.loc[end, "Datetime"]

    if len(gap_lengths) == 0:

        continue

    summary.append({

        "Pollutant": pollutant,

        "Missing Blocks": len(gap_lengths),

        "Average Gap": round(np.mean(gap_lengths), 2),

        "Median Gap": round(np.median(gap_lengths), 2),

        "Maximum Gap": np.max(gap_lengths),

        "95th Percentile": round(np.percentile(gap_lengths,95),2),

        "Longest Station": longest_station,

        "Gap Start": longest_start,

        "Gap End": longest_end

    })

gap_statistics = pd.DataFrame(summary)

print("\n")
print("=" * 80)
print("DETAILED GAP STATISTICS")
print("=" * 80)

print(gap_statistics)

print("\n")
print("=" * 80)
print("TOP LONGEST GAPS")
print("=" * 80)

print(
    gap_statistics.sort_values(
        by="Maximum Gap",
        ascending=False
    )
)

print("=" * 80)





# ============================================================
# Phase 15.3.2 : Gap Length Distribution Analysis
# ============================================================
#
# Purpose
# -------
# Visualize the distribution of remaining missing gap lengths.
#
# This helps determine the maximum gap length that can be
# reasonably imputed using a Kalman Filter.
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

print("\n")
print("=" * 80)
print("PHASE 15.3.2 : GAP LENGTH DISTRIBUTION")
print("=" * 80)

pollutant_columns = [
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
    "Toluene"
]

for pollutant in pollutant_columns:

    gap_lengths = []

    for station in aqi_df["StationId"].unique():

        station_df = (
            aqi_df[aqi_df["StationId"] == station]
            .sort_values("Datetime")
            .reset_index(drop=True)
        )

        is_missing = station_df[pollutant].isna()

        i = 0

        while i < len(station_df):

            if not is_missing.iloc[i]:
                i += 1
                continue

            start = i

            while i < len(station_df) and is_missing.iloc[i]:
                i += 1

            end = i - 1

            if start == 0:
                continue

            if end == len(station_df) - 1:
                continue

            gap_size = end - start + 1

            gap_lengths.append(gap_size)

    if len(gap_lengths) == 0:
        continue

    gap_lengths = np.array(gap_lengths)

    print("\n")
    print("-" * 60)
    print(f"{pollutant}")
    print("-" * 60)

    bins = {
        "7-12 Hours": np.sum((gap_lengths >= 7) & (gap_lengths <= 12)),
        "13-24 Hours": np.sum((gap_lengths >= 13) & (gap_lengths <= 24)),
        "25-48 Hours": np.sum((gap_lengths >= 25) & (gap_lengths <= 48)),
        "49-72 Hours": np.sum((gap_lengths >= 49) & (gap_lengths <= 72)),
        ">72 Hours": np.sum(gap_lengths > 72)
    }

    total = len(gap_lengths)

    for k, v in bins.items():

        print(f"{k:<15}: {v:5d} ({100*v/total:6.2f}%)")

    plt.figure(figsize=(8,5))

    plt.hist(
        gap_lengths,
        bins=40
    )

    plt.title(f"{pollutant} - Missing Gap Length Distribution")

    plt.xlabel("Gap Length (Hours)")

    plt.ylabel("Number of Missing Blocks")

    plt.grid(alpha=0.3)

    plt.show()

print("\n")
print("=" * 80)
print("Gap Length Distribution Analysis Completed")
print("=" * 80)


from  pykalman import KalmanFilter

MIN_GAP = 7 

MAX_GAP = 24 



def find_missing_blocks(series):

    blocks = []

    is_missing = series.isna()

    i = 0

    while i < len(series):

        # Skip valid observations
        if not is_missing.iloc[i]:
            i += 1
            continue

        start = i

        # Traverse the entire missing block
        while i < len(series) and is_missing.iloc[i]:
            i += 1

        end = i - 1

        # Ignore beginning gaps
        if start == 0:
            continue

        # Ignore ending gaps
        if end == len(series) - 1:
            continue

        previous = start - 1
        next_valid = end + 1

        blocks.append({

            "start": start,

            "end": end,

            "gap": end - start + 1,

            "previous": previous,

            "next": next_valid

        })

    return blocks



def kalman_estimate_series(series):

    values = series.astype(float).values

    # Convert NaNs into masked values
    masked_values = np.ma.masked_invalid(values)

    if masked_values.count() == 0:
        return values

    # Simple 1D Kalman Filter
    kf = KalmanFilter(

        transition_matrices=[1],

        observation_matrices=[1],

        initial_state_mean=masked_values[~masked_values.mask][0],

        transition_covariance=1,

        observation_covariance=4

    )

    estimated_states, _ = kf.smooth(masked_values)

    return estimated_states.flatten()




def kalman_impute_block(series,
                        min_gap = 7,
                        max_gap = 24):

    # Entire estimated series
    estimated = kalman_estimate_series(series)

    # Copy original series
    filled = series.copy()

    # Detect missing blocks
    blocks = find_missing_blocks(series)

    for block in blocks:

        gap = block["gap"]

        # Skip if not eligible
        if not (min_gap<= gap<=max_gap):
            continue

        # Replace ONLY missing values
        for idx in range(block["start"], block["end"] + 1):

            if pd.isna(filled.iloc[idx]):

                estimate = estimated[idx]

                # Safety check
                if estimate >= 0:

                    filled.iloc[idx] = estimate

    return filled


def kalman_station_imputation(
    dataframe,
    pollutant_columns,
    min_gap=7,
    max_gap=24
):
    """
    Apply Kalman imputation station-wise for every pollutant.

    Rules
    -----
    1. Process each station independently.
    2. Process each pollutant independently.
    3. Only fill gaps whose length is between
       min_gap and max_gap hours.
    4. Beginning/end gaps are never filled.
    5. Preserve every observed measurement.
    """

    df = dataframe.copy()

    print("\n")
    print("=" * 80)
    print("STARTING GENERIC KALMAN IMPUTATION")
    print("=" * 80)

    for pollutant in pollutant_columns:

        before_missing = df[pollutant].isna().sum()

        print("\n")
        print("-" * 70)
        print(f"Processing : {pollutant}")

        for station in df["StationId"].unique():

            station_mask = df["StationId"] == station

            station_df = (
                df.loc[station_mask]
                  .sort_values("Datetime")
                  .copy()
            )

            filled = kalman_impute_block(
                station_df[pollutant],
                min_gap=min_gap,
                max_gap=max_gap
            )

            station_df[pollutant] = filled

            df.loc[
                station_mask,
                pollutant
            ] = station_df[pollutant].values

        after_missing = df[pollutant].isna().sum()

        print(f"Missing Before : {before_missing:,}")
        print(f"Missing After  : {after_missing:,}")
        print(f"Recovered      : {before_missing-after_missing:,}")

    print("\n")
    print("=" * 80)
    print("GENERIC KALMAN IMPUTATION COMPLETED")
    print("=" * 80)

    return df




# ============================================================
# Phase 15.4 : Kalman Imputation (7–24 Hour Gaps)
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 15.4 : KALMAN IMPUTATION")
print("=" * 80)

aqi_df = kalman_station_imputation(
    dataframe=aqi_df,
    pollutant_columns=pollutant_columns,
    min_gap=7,
    max_gap=24
)

print("\n")
print("=" * 80)
print("PHASE 15.4 COMPLETED")
print("=" * 80)




# ============================================================
# PHASE 15.5 : FINAL MISSING VALUE MATRIX
# ============================================================

import missingno as msno
import matplotlib.pyplot as plt

print("\n")
print("=" * 80)
print("PHASE 15.5 : FINAL MISSING VALUE MATRIX")
print("=" * 80)

plt.figure(figsize=(18, 8))

msno.matrix(
    aqi_df,
    figsize=(18, 8),
    sparkline=True,
    fontsize=10
)

plt.title(
    "Missing Value Matrix After Interpolation + Kalman Imputation",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

print("=" * 80)
print("Final missing value matrix generated successfully.")
print("=" * 80)



# ============================================================
# Save Preprocessed Dataset
# ============================================================

aqi_df.to_csv(
    "preprocessed_aqi_dataset.csv",
    index=False
)

print("\n")
print("=" * 80)
print("PREPROCESSED DATASET SAVED SUCCESSFULLY")
print("Filename : preprocessed_aqi_dataset.csv")
print("=" * 80)
