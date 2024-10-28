import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

# Function to draw the map with typhoon paths (actual or predicted)
def draw_map(df, fig_type, ax, lat_min, lat_max, lon_min, lon_max):
    # Set map extent and features
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#CDE5F1')
    ax.add_feature(cfeature.OCEAN, facecolor='#65ACC8')
    
    # Draw gridlines
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Get longitude and latitude values
    x, y = df['LONG'].values, df['LAT'].values

    # Plot previous points
    sizes = [20 for _ in range(len(df)-1)]
    ax.scatter(x[:-1], y[:-1], s=sizes, color='#575153', marker='o', alpha=0.75, transform=ccrs.PlateCarree())

    # Plot last point with different style
    if df['I'].iloc[-1] != 9:
        s = 300 * (df['I'].iloc[-1] + 1)
        color = '#83bca4'
    else:
        s = 300
        color = '#e6dee2'
    ax.scatter(x[-1], y[-1], s=s, color=color, edgecolor='#895027', marker='o', alpha=0.9, transform=ccrs.PlateCarree())

    # Plot path line
    ax.plot(x, y, '-', markersize=1, linewidth=2, color='#9fdf4d', transform=ccrs.PlateCarree())
    
    # Add title with time, pressure, wind, and other data
    t, pres, wnd, owd = df['Time'].iloc[-1], df['PRES'].iloc[-1], df['WND'].iloc[-1], df['OWD'].iloc[-1]
    ax.set_title(f'Typhoon Path {fig_type}\n{t}\nPRES: {pres}hPa, WND: {wnd}m/s, OWD: {owd}m/s')

# Function to find the boundary for the map (latitude and longitude)
def find_boundary(df):
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})  # 1 row, 2 columns

    # Draw actual and predicted maps
    draw_map(actual_df, 'Actual_%d_%d' % (test_typhoonID, i), axes[0], lat_min, lat_max, lon_min, lon_max)
    draw_map(predict_df, 'Predict_%d_%d' % (test_typhoonID, i), axes[1], lat_min, lat_max, lon_min, lon_max)

    plt.tight_layout()
    plt.savefig('result/plot_%d_%d.jpg' % (test_typhoonID, i))  # Save the figure

def make_two_map_gif(all_actual_df, all_predict_df, test_typhoonID):
    # Get map boundaries for actual and predicted data
    a_lat_min, a_lon_min, a_lat_max, a_lon_max = find_boundary(all_actual_df)
    p_lat_min, p_lon_min, p_lat_max, p_lon_max = find_boundary(all_predict_df)

    # Set global boundaries to encompass both maps
    lat_min = min(a_lat_min, p_lat_min)
    lat_max = max(a_lat_max, p_lat_max)
    lon_min = min(a_lon_min, p_lon_min)
    lon_max = max(a_lon_max, p_lon_max)
    
    images = []
    # Draw maps for each unique index
    for i in all_actual_df['index'].unique():
        actual_df = all_actual_df[all_actual_df['index'] == i]
        predict_df = all_predict_df[all_predict_df['index'] == i]

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        draw_map(actual_df, 'Actual_%d_%d' % (test_typhoonID, i), axes[0], lat_min, lat_max, lon_min, lon_max)
        draw_map(predict_df, 'Predict_%d_%d' % (test_typhoonID, i), axes[1], lat_min, lat_max, lon_min, lon_max)

        plt.tight_layout()
        # Save figure to memory buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(Image.open(buf))
        plt.close()

    # Save images as animated GIF
    images[0].save('result/animation_%s.gif' % test_typhoonID, save_all=True, append_images=images[1:], duration=300, loop=0)