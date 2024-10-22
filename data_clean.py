from glob import glob
import pandas as pd
from share_func import create_dist_bear

buff = []
end = ''

# Iterate through all .txt files in the CMABSTdata folder
for txtfile in sorted(glob('CMABSTdata/*.txt')):
    data = open(txtfile).read().split('\n')  # Read and split the file by lines
    year = txtfile[8:12]  # Extract the year from the filename
    typhoon_id = ''
    
    for line in data:
        cols = line.split()  # Split each line by spaces
        if cols:
            if cols[0] == '66666':  # Special marker for new typhoon data
                typhoon_id = year + cols[3]  # Create a typhoon ID using the year and ID
                if len(buff) > 0:
                    buff[-1][-1] = end  # Update the END column for the previous record
                end = str(1 + int(cols[5]))  # Set the END flag based on the data
            else:
                if len(cols) < 7:
                    buff.append([typhoon_id] + cols + ['0', '0'])  # Fill missing values with 0
                else:
                    buff.append([typhoon_id] + cols + ['0'])  # Append the current line's data

# Create a DataFrame with the extracted data
df = pd.DataFrame(buff, columns=['typhoonID', 'Time', 'I', 'LAT', 'LONG', 'PRES', 'WND', 'OWD', 'END'])

# Convert the Time column to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H')

# Ensure specific columns are filled with 0 where missing and cast to integer
for col in ['I', 'LAT', 'LONG', 'PRES', 'WND', 'OWD', 'END']:
    df[col] = df[col].fillna(0).astype(int)

# Add Haversine distance and bearing columns to the DataFrame
df = create_dist_bear(df)

# Save the cleaned DataFrame to a CSV file
df.to_csv('cleaned_data.csv', index=False)