import numpy as np
import pandas as pd

def normalize(df, x):
    x_min = df[x].min()  # Get the minimum value of the column
    x_max = df[x].max()  # Get the maximum value of the column
    return (df[x] - x_min) / (x_max - x_min)  # Normalize to range [0, 1]

def denormalize(df, x, normalized_x):
    x_min = df[x].min()  # Get the minimum value of the column
    x_max = df[x].max()  # Get the maximum value of the column
    return normalized_x * (x_max - x_min) + x_min  # Restore to original scale

def normalize_df(df):
    norm_df = pd.DataFrame()
    norm_df['typhoonID'] = df['typhoonID']  # Preserve typhoonID

    # Sine and cosine for hour, month, and day cycles
    norm_df['hour_sin'] = np.sin(2 * np.pi * df['Time'].dt.hour / 24)
    norm_df['hour_cos'] = np.cos(2 * np.pi * df['Time'].dt.hour / 24)

    norm_df['month_sin'] = np.sin(2 * np.pi * df['Time'].dt.month / 12)
    norm_df['month_cos'] = np.cos(2 * np.pi * df['Time'].dt.month / 12)

    norm_df['day_sin'] = np.sin(2 * np.pi * df['Time'].dt.day / 31)
    norm_df['day_cos'] = np.cos(2 * np.pi * df['Time'].dt.day / 31)

    # Sine and cosine for day of year cycle
    df['day_of_year'] = df['Time'].dt.dayofyear
    norm_df['year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    norm_df['year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Normalize distance
    norm_df['distance_km'] = normalize(df, 'distance_km')

    # Sine and cosine for bearing
    norm_df['bearing_sin'] = np.sin(np.radians(df['bearing']))
    norm_df['bearing_cos'] = np.cos(np.radians(df['bearing']))

    # One-hot encoding for 'I' column
    df_one_hot_i = pd.get_dummies(df['I'], prefix='I')
    norm_df = pd.concat([norm_df, df_one_hot_i], axis=1)

    # Normalize PRES, WND, and OWD
    norm_df['pres_norm'] = normalize(df, 'PRES')
    norm_df['wnd_norm'] = normalize(df, 'WND')
    norm_df['owd_norm'] = normalize(df, 'OWD')

    #One-hot encoding for 'END' column
    df_one_hot_end = pd.get_dummies(df['END'], prefix='END')
    norm_df = pd.concat([norm_df, df_one_hot_end], axis=1)

    # Ensure all columns (except typhoonID) are float32
    norm_df = norm_df.astype({col: 'float32' for col in norm_df.columns[1:]})
    return norm_df

# Define Earth's radius in kilometers
R = 6371.0088  # km

# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

# Function to calculate bearing (direction)
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    # Calculate bearing and adjust to [0, 360]
    initial_bearing = np.arctan2(x, y)
    bearing = (np.degrees(initial_bearing) + 360) % 360
    return bearing


# Calculate new coordinates based on bearing and distance
def calculate_new_position(lat1, lon1, bearing, distance):
    # Convert latitude and longitude to radians (and divide by 10)
    lat1 = lat1 / 10
    lon1 = lon1 / 10
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    
    # Distance divided by Earth's radius
    angular_distance = distance / R
    bearing = np.radians(bearing)  # Convert bearing to radians

    # Calculate new latitude
    lat2 = np.arcsin(np.sin(lat1) * np.cos(angular_distance) + 
                     np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing))

    # Calculate new longitude
    lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
                             np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2))

    # Convert result back to degrees and multiply by 10
    lat2, lon2 = np.degrees(lat2), np.degrees(lon2)

    lat2 = lat2 * 10
    lon2 = lon2 * 10

    return lat2, lon2

def create_dist_bear(df):
    # Initialize two new columns for distance and bearing
    df['distance_km'] = [0.0] * len(df)
    df['bearing'] = [0.0] * len(df)

    # Calculate distance and bearing for each point relative to the previous point
    for i in range(1, len(df)):
        if df.loc[i - 1, 'END'] == 0:
            lat1, lon1 = df.loc[i - 1, 'LAT'], df.loc[i - 1, 'LONG']
            lat2, lon2 = df.loc[i, 'LAT'], df.loc[i, 'LONG']
            
            # Calculate distance
            df.loc[i, 'distance_km'] = haversine(lat1 / 10, lon1 / 10, lat2 / 10, lon2 / 10)
            
            # Calculate bearing
            df.loc[i, 'bearing'] = calculate_bearing(lat1 / 10, lon1 / 10, lat2 / 10, lon2 / 10)

    return df

def restore_bearing(bearing_sin, bearing_cos):
    # Restore bearing from sine and cosine values
    bearing_radians = np.arctan2(bearing_sin, bearing_cos)
    
    # Convert radians to degrees
    bearing_degrees = np.degrees(bearing_radians)
    
    # Ensure bearing is within [0, 360] degrees
    bearing_degrees = (bearing_degrees + 360) % 360
    
    return bearing_degrees