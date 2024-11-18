import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

offset1 = -0.3 # offset by degrees C, thermocouple calibration
offset2 = +0.03
offset3 = +0.35
offset4 = +0.45
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
# Plot temperatures
plt.figure(figsize=(10, 6))
for col in columns_to_plot:
    plt.plot(df['Time_seconds'], df[col], label=col)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.title('Temperature vs. Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot heat fluxes
def calculate_sensitivity(S_0, S_C, T_S, T_0=22.5):
    return S_0 + (T_S - T_0) * S_C

def calculate_heatflux_vectorized(U, S):
    return U.astype(float)*(1000*1000) / S

calibration_data_file_path = 'Heat_flux_sensors_calibration.csv'

calibration_data = pd.read_csv(calibration_data_file_path, sep=';')
calibration_data['Sensitivity S0'] = pd.to_numeric(calibration_data['Sensitivity S0'], errors='coerce')
calibration_data['Correction factor Sc'] = pd.to_numeric(calibration_data['Correction factor Sc'], errors='coerce')
HeatfluxDatavoltage = {}
HeatfluxData = {}
Heatfluxcolumns = ['E1 flux Last (V)', 'E2 flux Last (V)']
for col in Heatfluxcolumns:
    HeatfluxDatavoltage[col] = df[col]

for column in Heatfluxcolumns:
    if column == "E1 flux Last (V)":
        sensor_id = 'E1'
        calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
        S_0 = calibration_row['Sensitivity S0']
        S_C = calibration_row['Correction factor Sc']
        S = calculate_sensitivity(S_0, S_C, T_S=15)

        HeatfluxData['E1 flux Last (V)'] = calculate_heatflux_vectorized(HeatfluxDatavoltage['E1 flux Last (V)'], S)
        # #print(column[0:2])
        # calibration_row = calibration_data[calibration_data['serial number'] == sensor_id]
        # HeatfluxData['HeatFlux' + column] = sensor_data.apply(
        # lambda row: calculate_heatflux(row[column], T_S, calibration_row['number'].values[0], calibration_data),
        # axis=1
        # )
    if column == "E2 flux Last (V)":
        sensor_id = 'E2'
        calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
        S_0 = calibration_row['Sensitivity S0']
        S_C = calibration_row['Correction factor Sc']
        S = calculate_sensitivity(S_0, S_C, T_S=15)

        HeatfluxData['E2 flux Last (V)'] = calculate_heatflux_vectorized(HeatfluxDatavoltage['E2 flux Last (V)'], S)
for col in Heatfluxcolumns:
    plt.plot(df['Time_seconds'], savgol_filter(HeatfluxData[col], window_length=3000, polyorder=2), label=col)
plt.xlabel('Time (s)')
plt.ylabel('Heat Flux (W/m^2)')
plt.title('Heat Flux vs. Time')
plt.legend()
plt.grid(True)
plt.show()