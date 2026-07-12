import pandas as pd
import numpy as np 

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
# 3. Interpolate ONLY if elapsed time &lt;= MAX_GAP_HOURS.
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
# ------------------------------------------------------------
# Create Synthetic Dataset
# ------------------------------------------------------------
#
# Test Cases
# ----------
#
# STA
# ----
# Small gap (2 hours)
# Expected:
# 100, 110, 120, 130
#
# STB
# ----
# Elapsed gap = 6 hours
# Should interpolate.
#
# STC
# ----
# Elapsed gap = 7 hours
# Should NOT interpolate.
#
# STD
# ----
# Missing values at beginning.
# Should remain NaN.
#
# STE
# ----
# Missing values at end.
# Should remain NaN.
# ------------------------------------------------------------

data = {

    "StationId":[

        # ---------------- STA ----------------
        "STA","STA","STA","STA",

        # ---------------- STB ----------------
        "STB","STB","STB","STB","STB","STB","STB",

        # ---------------- STC ----------------
        "STC","STC","STC","STC","STC","STC","STC","STC",

        # ---------------- STD ----------------
        "STD","STD","STD","STD",

        # ---------------- STE ----------------
        "STE","STE","STE","STE"

    ],

    "Datetime":[

        # ====================================================
        # STA
        # ====================================================

        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 02:00",
        "2024-01-01 03:00",

        # ====================================================
        # STB
        # Previous = 00:00
        # Next = 06:00
        # Elapsed = 6 Hours
        # ====================================================

        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 02:00",
        "2024-01-01 03:00",
        "2024-01-01 04:00",
        "2024-01-01 05:00",
        "2024-01-01 06:00",

        # ====================================================
        # STC
        # Previous = 00:00
        # Next = 07:00
        # Elapsed = 7 Hours
        # ====================================================

        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 02:00",
        "2024-01-01 03:00",
        "2024-01-01 04:00",
        "2024-01-01 05:00",
        "2024-01-01 06:00",
        "2024-01-01 07:00",

        # ====================================================
        # STD
        # Missing at beginning
        # ====================================================

        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 02:00",
        "2024-01-01 03:00",

        # ====================================================
        # STE
        # Missing at end
        # ====================================================

        "2024-01-01 00:00",
        "2024-01-01 01:00",
        "2024-01-01 02:00",
        "2024-01-01 03:00"

    ],

    "PM2.5":[

        # ====================================================
        # STA
        # Should become:
        # 100,110,120,130
        # ====================================================

        100,
        np.nan,
        np.nan,
        130,

        # ====================================================
        # STB
        # Elapsed gap = 6 hours
        # SHOULD interpolate
        # ====================================================

        200,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        260,

        # ====================================================
        # STC
        # Elapsed gap = 7 hours
        # SHOULD NOT interpolate
        # ====================================================

        300,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        370,

        # ====================================================
        # STD
        # Missing at beginning
        # SHOULD remain NaN
        # ====================================================

        np.nan,
        np.nan,
        500,
        510,

        # ====================================================
        # STE
        # Missing at end
        # SHOULD remain NaN
        # ====================================================

        600,
        610,
        np.nan,
        np.nan

    ]

}

test_df = pd.DataFrame(data)

# Convert datetime

test_df["Datetime"] = pd.to_datetime(test_df["Datetime"])

print("="*60)
print("Synthetic Test Dataset")
print("="*60)

print(test_df)


# ============================================================
# Phase 2 : Detect Missing Blocks
# ============================================================
#
# Objective
# ---------
# Detect every consecutive missing-value block for each station.
#
# No interpolation is performed in this step.
#
# We only report:
#   • Previous valid timestamp
#   • Next valid timestamp
#   • Elapsed time
#   • Whether interpolation is allowed
#
# Rule:
# -----
# Interpolate ONLY if:
#
# elapsed_time &lt;= 6 hours
# ============================================================

MAX_GAP_HOURS = 6

print("\n")
print("=" * 80)
print("MISSING BLOCK DETECTION")
print("=" * 80)

# ------------------------------------------------------------
# Process each station independently
# ------------------------------------------------------------

for station, station_df in test_df.groupby("StationId"):

    station_df = station_df.sort_values("Datetime").reset_index(drop=True)

    series = station_df["PM2.5"]

    print(f"\nStation : {station}")
    print("-" * 60)

    i = 0

    while i < len(series):

        # ----------------------------------------------------
        # Skip non-missing values
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Check whether block has valid neighbours
        # ----------------------------------------------------

        if start == 0:

            print("Missing block at beginning -&gt; NOT interpolated")

            continue

        if end == len(series) - 1:

            print("Missing block at end -&gt; NOT interpolated")

            continue

        prev_time = station_df.loc[start - 1, "Datetime"]
        next_time = station_df.loc[end + 1, "Datetime"]

        elapsed_hours = (
            next_time - prev_time
        ).total_seconds() / 3600

        print(f"Missing Block : {start} -&gt; {end}")

        print(f"Previous Time : {prev_time}")

        print(f"Next Time     : {next_time}")

        print(f"Elapsed Hours : {elapsed_hours}")

        if elapsed_hours <= MAX_GAP_HOURS:

            print("Decision      : INTERPOLATE")

        else:

            print("Decision      : KEEP AS NaN")

print("\n")
print("=" * 80)
print("Gap Detection Completed Successfully")
print("=" * 80)




# ============================================================
# Phase 4 : Unit Test for interpolate_block()
# ============================================================

print("\n")
print("=" * 80)
print("UNIT TEST : interpolate_block()")
print("=" * 80)

previous_value = 100
next_value = 130

previous_time = pd.Timestamp("2024-01-01 00:00:00")
next_time = pd.Timestamp("2024-01-01 03:00:00")

missing_times = [

    pd.Timestamp("2024-01-01 01:00:00"),

    pd.Timestamp("2024-01-01 02:00:00")

]

result = interpolate_block(

    previous_value,
    next_value,
    previous_time,
    next_time,
    missing_times

)

print("Interpolated Values")

for value in result:

    print(round(value, 2))

expected = [110.0, 120.0]

assert np.allclose(result, expected)

print("\n")
print("PASS")
print("Interpolation function works correctly.")
print("=" * 80)

# ============================================================
# Phase 6 : Test Generic Interpolation Engine
# ============================================================

print("\n")
print("=" * 80)
print("TESTING GENERIC INTERPOLATION ENGINE")
print("=" * 80)

# ------------------------------------------------------------
# Apply interpolation station by station
# ------------------------------------------------------------

result = (
    test_df
    .groupby("StationId")
    .apply(
        lambda station: interpolate_station(
            station,
            "PM2.5"
        )
    )
)

# ------------------------------------------------------------
# Convert the MultiIndex back into normal columns.
#
# This guarantees that StationId becomes a dataframe column
# regardless of the pandas version being used.
# ------------------------------------------------------------

result = (
    result
    .reset_index(level=0)
    .rename(columns={"level_0": "StationId"})
    .reset_index(drop=True)
)

print(result)

print("\n")
print("=" * 80)
print("Generic Interpolation Engine Passed")
print("=" * 80)

print(result.index)
print(result.columns)
 # ============================================================
# Phase 7 : Automatic Unit Tests
# ============================================================

print("\n")
print("=" * 80)
print("RUNNING AUTOMATED UNIT TESTS")
print("=" * 80)

# ------------------------------------------------------------
# Test Case 1 : STA
# ------------------------------------------------------------

sta = result[result["StationId"] == "STA"]["PM2.5"].reset_index(drop=True)

expected_sta = pd.Series([100.0, 110.0, 120.0, 130.0])

pd.testing.assert_series_equal(
    sta,
    expected_sta,
    check_names=False
)

print("PASS : STA interpolation")

# ------------------------------------------------------------
# Test Case 2 : STB
# ------------------------------------------------------------

stb = result[result["StationId"] == "STB"]["PM2.5"].reset_index(drop=True)

expected_stb = pd.Series([
    200.0,
    210.0,
    220.0,
    230.0,
    240.0,
    250.0,
    260.0
])

pd.testing.assert_series_equal(
    stb,
    expected_stb,
    check_names=False
)

print("PASS : STB interpolation")

# ------------------------------------------------------------
# Test Case 3 : STC
# ------------------------------------------------------------

stc = result[result["StationId"] == "STC"]["PM2.5"].reset_index(drop=True)

assert stc.isna().sum() == 6

print("PASS : STC large-gap protection")

# ------------------------------------------------------------
# Test Case 4 : STD
# ------------------------------------------------------------

std = result[result["StationId"] == "STD"]["PM2.5"].reset_index(drop=True)

assert pd.isna(std.iloc[0])
assert pd.isna(std.iloc[1])

print("PASS : STD beginning-gap protection")

# ------------------------------------------------------------
# Test Case 5 : STE
# ------------------------------------------------------------

ste = result[result["StationId"] == "STE"]["PM2.5"].reset_index(drop=True)

assert pd.isna(ste.iloc[2])
assert pd.isna(ste.iloc[3])

print("PASS : STE ending-gap protection")

# ------------------------------------------------------------
# Final Success Message
# ------------------------------------------------------------

print("\n")
print("=" * 80)
print("ALL UNIT TESTS PASSED")
print("Interpolation Engine Successfully Validated")
print("=" * 80)