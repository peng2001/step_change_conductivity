import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('your_file.csv')

# Rename the unnamed time column to 'Time'
df.rename(columns={df.columns[0]: 'Time'}, inplace=True)

# Convert the 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'])

# Calculate the time in seconds from the first timestamp
df['Time_seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

# Plot the temperatures as a function of time
plt.figure(figsize=(10, 6))

for col in df.columns[1:6]:  # Assuming the first 5 temperature columns are next to the time column
    plt.plot(df['Time_seconds'], df[col], label=col)

plt.xlabel('Time (seconds from the start)')
plt.ylabel('Temperature (C)')
plt.title('Temperature vs. Time')
plt.legend()
plt.grid(True)
plt.show()
