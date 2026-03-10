import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress

# --- Constants ---
INPUT_DIR = 'input'
OUTPUT_DIR = 'plots'

# --- Heat Index Formula (Simplified version for temp in Celsius and RH) ---
def compute_heat_index(temp_c, rh):
    # Convert C to F
    T = temp_c * 9/5 + 32
    HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (rh * 0.094))
    if HI >= 80:
        HI = -42.379 + 2.04901523 * T + 10.14333127 * rh \
             - 0.22475541 * T * rh - 0.00683783 * T ** 2 \
             - 0.05481717 * rh ** 2 + 0.00122874 * T ** 2 * rh \
             + 0.00085282 * T * rh ** 2 - 0.00000199 * T ** 2 * rh ** 2
    return (HI - 32) * 5/9  # Convert back to Celsius

# --- Load File ---
import os
import json

# List all JSON files in the input folder
INPUT_DIR = 'input'
json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]

# Display numbered options
print("Available JSON files:")
for idx, filename in enumerate(json_files, 1):
    print(f"{idx}. {filename}")

# Prompt user for selection
choice = int(input("Enter the number of the file to process: ")) - 1

# Load the selected JSON file
file_path = os.path.join(INPUT_DIR, json_files[choice])
with open(file_path, 'r') as f:
    data = json.load(f)

print(f"Loaded file: {json_files[choice]}")

with open(file_path, 'r') as f:
    data = json.load(f)

# --- Extract Data ---
t2m = data['properties']['parameter']['T2M']
rh2m = data['properties']['parameter']['RH2M']

dates = sorted(set(t2m.keys()) & set(rh2m.keys()))
records = []

for date_str in dates:
    date = datetime.strptime(date_str, '%Y%m%d')
    temp = t2m[date_str]
    rh = rh2m[date_str]
    hi = compute_heat_index(temp, rh)
    records.append({'date': date, 'temperature': temp, 'humidity': rh, 'HI': hi})

df = pd.DataFrame(records)

# --- Save to JSON ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Extract filename without extension
base_name = os.path.splitext(json_files[choice])[0]

# Create output file paths
output_json = os.path.join(OUTPUT_DIR, f'{base_name}_HI_result.json')
output_plot = os.path.join(OUTPUT_DIR, f'{base_name}_HI_plot.png')

# Save JSON
df[['date', 'HI']].to_json(output_json, orient='records', date_format='iso')


# --- Plotting ---
plt.figure(figsize=(18, 5))
plt.plot(df['date'], df['HI'], label='Daily HI', linewidth=1)

# Trendline
x = np.arange(len(df))
slope, intercept, *_ = linregress(x, df['HI'])
plt.plot(df['date'], intercept + slope * x, 'r--', label='Trendline')

# Set axis limits
plt.ylim(0, 100)  # Y-axis: HI in Celsius from 0 to 100
plt.title('Daily Heat Index (HI)')
plt.xlabel('Date')
plt.ylabel('HI (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot)  # ✅ dynamic filename e.g., AnandVihar_HI_plot.png
plt.show()