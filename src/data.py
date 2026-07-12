import pandas as pd 

import matplotlib.pyplot as plt 

df = pd.read_csv(r"/Users/sohan/Documents/AI-Based-Air-Quality-Prediction-System/data/raw/station_hour.csv")

print(df.head())

print(df.tail())

print(df["AQI"].mean())

plt.figure(figsize=(12,6))

plt.hist(df[df["AQI"] <= 600]["AQI"] , bins = 60 , edgecolor = 'black')
plt.xlim(0,600)

plt.grid(alpha = 0.3)

#plt.show()

df_ = pd.read_csv(r"/Users/sohan/Documents/AI-Based-Air-Quality-Prediction-System/data/raw/stations.csv")

print(df_.columns)
print(df_.head())

merged_df = pd.merge(df , df_ , on = "StationId" , how="left")

print(merged_df["City"].value_counts())

top5 = ["Delhi", "Bengaluru", "Hyderabad", "Chennai", "Mumbai"]

for city in top5:
    city_df = merged_df[merged_df["City"] == city]
    print(f"\n{city}")
    print(city_df.isnull().mean() * 100)


    # -------------------------------------------------------------
# Selecting the Two Cities for Model Development
# -------------------------------------------------------------
# Based on exploratory data analysis (EDA), we selected only
# Delhi and Hyderabad for the following reasons:
#
# 1. Delhi has the highest number of hourly observations,
#    providing a rich time-series dataset for model training.
#
# 2. Hyderabad has the second-best overall data quality among
#    the major cities, with significantly fewer missing values
#    than Bengaluru, Chennai, and Mumbai for important pollutant
#    parameters.
#
# 3. Using these two cities provides a large, high-quality
#    dataset while reducing preprocessing complexity and
#    computational cost.
#
# This selection follows the mentor's recommendation to work
# with two cities having sufficient data for accurate AQI
# prediction.
# -------------------------------------------------------------

# Keep only Delhi and Hyderabad records
selected_cities = ["Delhi", "Hyderabad"]

filtered_df = merged_df[merged_df["City"].isin(selected_cities)].copy()

# Verify the remaining cities
print("Selected Cities:")
print(filtered_df["City"].value_counts())

# Check the size of the filtered dataset
print("\nFiltered Dataset Shape:", filtered_df.shape)



# Save the filtered dataset for the next stages of the project
filtered_df.to_csv("Delhi_Hyderabad_Station_Hour.csv", index=False)

print("Filtered dataset saved successfully.")