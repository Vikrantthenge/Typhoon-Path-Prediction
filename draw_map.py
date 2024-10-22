import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

# Function to draw the map with actual or predicted typhoon paths
def draw_map(df, fig_type, ax, lat_min, lat_max, lon_min, lon_max):
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max, 
                llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='coral', lake_color='aqua')

    # Draw latitude (parallels) and longitude (meridians) lines
    parallels = np.arange(lat_min, lat_max, 1)  
    meridians = np.arange(lon_min, lon_max, 1)  

    # Draw latitude lines with labels
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)

    # Draw longitude lines with labels
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

    # Plot the typhoon path
    x, y = m(df['LONG'].values, df['LAT'].values)
    sizes = [100 * (i+1) for i in df['I']]  # Adjust point sizes based on intensity
    m.scatter(x, y, s=sizes, color='b', marker='o', alpha=0.75)  # Scatter plot for size
    m.plot(x, y, 'D-', markersize=1, linewidth=2, color='b', markerfacecolor='b')

    # Annotate the time, pressure, wind, and other data points
    for t, pres, wnd, owd, xpt, ypt in zip(df['Time'], df['PRES'], df['WND'], df['OWD'], x, y):
        ax.text(xpt + 10000, ypt - 5000, f'{t}, PRES: {pres}hPa,\nWND: {wnd}m/s, OWD: {owd}m/s', 
                 fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

    ax.set_title('Typhoon Path %s' % fig_type)

# Function to find the boundary for the map (latitude and longitude)
def find_boundary(df):
    # Convert LAT and LONG to proper format
    df['LAT'] = df['LAT'] / 10
    df['LONG'] = df['LONG'] / 10
    lat_min, lat_max = df['LAT'].min(), df['LAT'].max()
    lon_min, lon_max = df['LONG'].min(), df['LONG'].max()

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # Add a 20% margin to the boundary
    lat_min -= lat_range * 0.2
    lat_max += lat_range * 0.2
    lon_min -= lon_range * 0.2
    lon_max += lon_range * 0.2

    lat_min = np.floor(lat_min)
    lon_min = np.floor(lon_min)
    lat_max = np.ceil(lat_max)
    lon_max = np.ceil(lon_max)

    return lat_min, lon_min, lat_max, lon_max

# Function to draw two maps side by side (actual vs predicted typhoon path)
def draw_two_map(actual_df, predict_df, test_typhoonID, i):
    # Get map boundaries for both actual and predicted data
    a_lat_min, a_lon_min, a_lat_max, a_lon_max = find_boundary(actual_df)
    p_lat_min, p_lon_min, p_lat_max, p_lon_max = find_boundary(predict_df)

    # Set global boundaries covering both maps
    lat_min = min(a_lat_min, p_lat_min)
    lat_max = max(a_lat_max, p_lat_max)
    lon_min = min(a_lon_min, p_lon_min)
    lon_max = max(a_lon_max, p_lon_max)

    # Create the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Draw actual and predicted maps
    draw_map(actual_df, 'Actual_%d_%d' % (test_typhoonID, i), axes[0], lat_min, lat_max, lon_min, lon_max)
    draw_map(predict_df, 'Predict_%d_%d' % (test_typhoonID, i), axes[1], lat_min, lat_max, lon_min, lon_max)

    plt.tight_layout()
    plt.savefig('result/plot_%d_%d.jpg' % (test_typhoonID, i))  # Save the figure