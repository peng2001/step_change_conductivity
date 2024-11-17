import pandas as pd
import matplotlib.pyplot as plt

offset1 = 0 # offset by degrees C, thermocouple calibration
offset2 = 0
offset3 = 0
offset4 = 0
offset5 = 0

# Load the CSV file
df = pd.read_csv('data/in-plane-thermal-cond/no_current_wires_data/100SOC-asym.csv')

# Rename the unnamed time column to 'Time'
df.rename(columns={df.columns[0]: 'Time'}, inplace=True)

# Convert the 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'])

# Calculate the time in seconds from the first timestamp
df['Time_seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

# Specify the columns you want to plot and their corresponding offsets
columns_to_plot = ['T1 Last (C)', 'T2 Last (C)', 'T3 Last (C)', 'T4 Last (C)', 'T5 Last (C)']
offsets = [offset1, offset2, offset3, offset4, offset5]

# Apply the offsets to the respective columns
for col, offset in zip(columns_to_plot, offsets):
    df[col] = df[col] + offset

# Plot the specified temperatures as a function of time
plt.figure(figsize=(10, 6))

for col in columns_to_plot:
    plt.plot(df['Time_seconds'], df[col], label=col)

plt.xlabel('Time (seconds from the start)')
plt.ylabel('Temperature (C)')
plt.title('Temperature vs. Time')
plt.legend()
plt.grid(True)
plt.show()
