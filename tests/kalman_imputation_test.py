# ============================================================
# Kalman Imputation Test
# ============================================================
#
# Phase 1 : Synthetic Dataset
#
# Purpose
# -------
# Create a synthetic hourly air-quality dataset containing
# multiple missing-value scenarios for validating the Kalman
# imputation engine.
#
# The dataset includes:
#
# STA : 8-hour gap       (Should be filled)
# STB : 12-hour gap      (Should be filled)
# STC : 24-hour gap      (Should be filled)
# STD : 25-hour gap      (Should NOT be filled)
# STE : Missing at start (Should NOT be filled)
# STF : Missing at end   (Should NOT be filled)
# STG : No missing data  (Should remain unchanged)
# STH : Two gaps
#       - 8 hours        (Should be filled)
#       - 30 hours       (Should NOT be filled)
#
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------

def create_station(station_id, values):

    return pd.DataFrame({

        "StationId": station_id,

        "Datetime": pd.date_range(
            start="2024-01-01 00:00:00",
            periods=len(values),
            freq="h"
        ),

        "PM2.5": values

    })

# ============================================================
# STA
# 8-hour gap
# ============================================================

STA = create_station(

    "STA",

    [
        100,
        102,
        104,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        126,
        128,
        130
    ]

)

# ============================================================
# STB
# 12-hour gap
# ============================================================

STB = create_station(

    "STB",

    [
        200,
        202,
        204,

        np.nan,np.nan,np.nan,np.nan,
        np.nan,np.nan,np.nan,np.nan,
        np.nan,np.nan,np.nan,np.nan,

        230,
        232
    ]

)

# ============================================================
# STC
# 24-hour gap
# ============================================================

stc_values = [300,302]

stc_values.extend([np.nan]*24)

stc_values.extend([350,352])

STC = create_station(

    "STC",

    stc_values

)

# ============================================================
# STD
# 25-hour gap
# ============================================================

std_values = [400,402]

std_values.extend([np.nan]*25)

std_values.extend([455,458])

STD = create_station(

    "STD",

    std_values

)

# ============================================================
# STE
# Missing at beginning
# ============================================================

STE = create_station(

    "STE",

    [

        np.nan,
        np.nan,
        np.nan,

        500,
        503,
        506,
        509,
        512

    ]

)

# ============================================================
# STF
# Missing at end
# ============================================================

STF = create_station(

    "STF",

    [

        600,
        603,
        606,
        609,

        np.nan,
        np.nan,
        np.nan,
        np.nan

    ]

)

# ============================================================
# STG
# No missing values
# ============================================================

STG = create_station(

    "STG",

    [

        700,
        702,
        704,
        706,
        708,
        710,
        712,
        714

    ]

)

# ============================================================
# STH
# Two gaps
# ============================================================

sth_values = [

    800,
    802,
    804

]

# 8-hour gap

sth_values.extend([np.nan]*8)

sth_values.extend([830,832])

# 30-hour gap

sth_values.extend([np.nan]*30)

sth_values.extend([900,902])

STH = create_station(

    "STH",

    sth_values

)

# ============================================================
# Combine Dataset
# ============================================================

test_df = pd.concat(

    [

        STA,
        STB,
        STC,
        STD,
        STE,
        STF,
        STG,
        STH

    ],

    ignore_index=True

)

# ============================================================
# Display Dataset
# ============================================================

print("="*80)

print("PHASE 1 : SYNTHETIC DATASET")

print("="*80)

print()

print(test_df)

print()

print("="*80)

print("Total Rows :", len(test_df))

print("Stations   :", test_df["StationId"].nunique())

print("="*80)




# ============================================================
# Phase 2 : Missing Block Detection
# ============================================================
#
# Purpose
# -------
# Detect every valid missing block inside a time series.
#
# Beginning gaps and ending gaps are ignored because they
# cannot be estimated reliably.
#
# Returns
# -------
# A list of dictionaries containing:
#
# start      -&gt; First missing index
# end        -&gt; Last missing index
# gap        -&gt; Gap length
# previous   -&gt; Previous valid index
# next       -&gt; Next valid index
# ============================================================

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



# ============================================================
# Phase 2 Test
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 2 : MISSING BLOCK DETECTION")
print("=" * 80)

for station in test_df["StationId"].unique():

    print("\n")
    print("-" * 60)
    print(f"Station : {station}")
    print("-" * 60)

    station_df = (
        test_df[test_df["StationId"] == station]
        .reset_index(drop=True)
    )

    blocks = find_missing_blocks(station_df["PM2.5"])

    if len(blocks) == 0:

        print("No valid missing blocks detected.")

    else:

        for block in blocks:

            print(
                f"Gap : {block['gap']:2d} hours | "
                f"Start : {block['start']:2d} | "
                f"End : {block['end']:2d}"
            )

print("\n")
print("=" * 80)
print("Phase 2 Completed Successfully")
print("=" * 80)



# ============================================================
# Phase 3A : Kalman State Estimation
# ============================================================
#
# Purpose
# -------
# Estimate the complete pollutant time series using a Kalman
# Filter.
#
# This function DOES NOT modify the original data.
# It only returns the estimated series.
# ============================================================

from pykalman import KalmanFilter


def kalman_estimate_series(series):

    values = series.astype(float).values

    # Convert NaNs into masked values
    masked_values = np.ma.masked_invalid(values)

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


# ============================================================
# Phase 3A Test
# ============================================================

print("\n")
print("=" * 80)
print("PHASE 3A : KALMAN STATE ESTIMATION")
print("=" * 80)

station_df = (
    test_df[test_df["StationId"] == "STA"]
    .reset_index(drop=True)
)

estimated = kalman_estimate_series(
    station_df["PM2.5"]
)

comparison = pd.DataFrame({

    "Datetime": station_df["Datetime"],

    "Original": station_df["PM2.5"],

    "Kalman Estimate": np.round(estimated,2)

})

print(comparison)

print("\n")
print("=" * 80)
print("Phase 3A Completed")
print("=" * 80)



# ============================================================
# Phase 3B : Kalman Imputation
# ============================================================
#
# Purpose
# -------
# Replace ONLY missing values belonging to
# eligible gaps (7-24 hours).
#
# Observed values are NEVER modified.
#
# ============================================================

MIN_GAP = 7
MAX_GAP = 24


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

# ============================================================
# Phase 5 : Generic Station-wise Kalman Imputation
# ============================================================

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
# Phase 3B Test
# ============================================================
results = {}
print("\n")
print("=" * 80)
print("PHASE 3B : KALMAN IMPUTATION")
print("=" * 80)

for station in test_df["StationId"].unique():

    print("\n")
    print("-" * 70)
    print(f"Station : {station}")
    print("-" * 70)

    station_df = (
        test_df[test_df["StationId"] == station]
        .reset_index(drop=True)
    )

    filled = kalman_impute_block(
        station_df["PM2.5"]
    )

    comparison = pd.DataFrame({

        "Datetime": station_df["Datetime"],

        "Original": station_df["PM2.5"],

        "After Kalman": np.round(filled,2)

    })

    print(comparison)

    comparison = station_df.copy()

    comparison["PM2.5"] = filled

    results[station] = comparison   


print("\n")
print("=" * 80)
print("RESULTS DICTIONARY")
print("=" * 80)

print(results.keys())
print("Stations Stored:", len(results))




# PHASE 4 : AUTOMATED UNIT TESTS
# ============================================================

import numpy as np

print("\n")
print("=" * 80)
print("PHASE 4 : AUTOMATED UNIT TESTS")
print("=" * 80)


# ------------------------------------------------------------
# STA
# 8-hour gap should be filled
# ------------------------------------------------------------

sta = results["STA"]

assert sta["PM2.5"].isna().sum() == 0

print("PASS : STA")


# ------------------------------------------------------------
# STB
# 12-hour gap should be filled
# ------------------------------------------------------------

stb = results["STB"]

assert stb["PM2.5"].isna().sum() == 0

print("PASS : STB")


# ------------------------------------------------------------
# STC
# 24-hour gap should be filled
# ------------------------------------------------------------

stc = results["STC"]

assert stc["PM2.5"].isna().sum() == 0

print("PASS : STC")


# ------------------------------------------------------------
# STD
# 25-hour gap should NOT be filled
# ------------------------------------------------------------

std = results["STD"]

assert std["PM2.5"].isna().sum() == 25

print("PASS : STD")


# ------------------------------------------------------------
# STE
# Beginning gap should remain NaN
# ------------------------------------------------------------

ste = results["STE"]

assert pd.isna(ste.iloc[0]["PM2.5"])
assert pd.isna(ste.iloc[1]["PM2.5"])
assert pd.isna(ste.iloc[2]["PM2.5"])

print("PASS : STE")


# ------------------------------------------------------------
# STF
# Ending gap should remain NaN
# ------------------------------------------------------------

stf = results["STF"]

assert pd.isna(stf.iloc[-1]["PM2.5"])
assert pd.isna(stf.iloc[-2]["PM2.5"])
assert pd.isna(stf.iloc[-3]["PM2.5"])
assert pd.isna(stf.iloc[-4]["PM2.5"])

print("PASS : STF")


# ------------------------------------------------------------
# STG
# No missing values anywhere
# ------------------------------------------------------------

stg = results["STG"]

assert stg["PM2.5"].isna().sum() == 0

print("PASS : STG")


# ------------------------------------------------------------
# STH
# First gap filled
# Second gap untouched
# ------------------------------------------------------------

sth = results["STH"]

first_gap = sth.iloc[3:11]["PM2.5"]

assert first_gap.isna().sum() == 0

second_gap = sth.iloc[13:43]["PM2.5"]

assert second_gap.isna().sum() == 30

print("PASS : STH")


print("\n")
print("=" * 80)
print("ALL KALMAN UNIT TESTS PASSED")
print("Kalman Imputation Engine Successfully Validated")
print("=" * 80)






print("\n")
print("=" * 80)
print("PHASE 5 : TESTING GENERIC KALMAN ENGINE")
print("=" * 80)

pollutants = ["PM2.5"]

generic_result = kalman_station_imputation(
    test_df,
    pollutants,
    min_gap=7,
    max_gap=24
)

print("\n")
print(generic_result.head(30))