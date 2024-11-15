directory = 'data/in-plane-thermal-cond/no_current_wires_data'

import os
import pandas as pd
import numpy as np
from io import StringIO

data_columns_Peltier = ['Current_temp_1', 'Current_temp_2', 'time']

def calculate_sensitivity(S_0, S_C, T_S, T_0=22.5):
    return S_0 + (T_S - T_0) * S_C

def calculate_heatflux_vectorized(U, S):
    return U.astype(float) / S

# List all files in the directory
files = os.listdir(directory)
files = sorted(files, key=lambda x: (not 'peltier_control' in x, x)) # first evaluate peltier_control

for file in files:
    if 'eric_carlos' in file:
        # Determine the prefix (A, B, C, or D) from the filename
        prefix = file.split('_')[2]
        
        # Construct the variable name and read the CSV file
        variable_name = f"Peltier_control"
        filepath = os.path.join(directory, file)
        
        # Read the CSV file into a pandas DataFrame
        locals()[variable_name] = pd.read_csv(filepath, sep=',')
        
        # Convert unix time stamp to datetime
        locals()[variable_name]['time'] = pd.to_datetime(locals()[variable_name][' unix_time_stamp'], unit='s')

        columns_to_keep = [col for col in locals()[variable_name].columns if col in data_columns_Peltier]
        locals()[variable_name] = locals()[variable_name][columns_to_keep]

    elif file.endswith('.txt'):
        txt_file_path = os.path.join(directory, file)
        
        # Read the file content
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
        
        # Separate metadata and data sections
        metadata_lines = []
        data_lines = []
        data_section = False
        
        for line in lines:
            if not data_section:
                metadata_lines.append(line.strip())
                if line.strip().startswith('Rec'):
                    data_section = True
                    data_lines.append(line.strip())
            else:
                data_lines.append(line.strip())
        
        # Process metadata
        metadata = {}
        for line in metadata_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        
        # Create DataFrame for data section
        data = StringIO('\n'.join(data_lines))
        CyclerData = pd.read_csv(data, sep='\t')

        CyclerData['MD'] = CyclerData['MD'].astype('category')
        CyclerData.loc[CyclerData['MD'] == 'D', 'Current'] *= -1
        
        # Convert the 'DPT Time' column to a datetime object
        CyclerData['DPT Time'] = pd.to_datetime(CyclerData['DPT Time'], format='%d-%b-%y %I:%M:%S %p') 
        CyclerData['DPT Time'] = CyclerData['DPT Time'].dt.tz_localize('Europe/London')

    elif 'stepchange' in file:
        sensor_data_file_path = os.path.join(directory, file)

        sensor_data = pd.read_csv(sensor_data_file_path, decimal=",")
        sensor_data = sensor_data[['Unnamed: 0', 'E1 flux Last (V)', 'E2 flux Last (V)']]
        # sensor_data = sensor_data.apply(pd.to_numeric, errors='coerce')
        sensor_data.columns = sensor_data.columns.str.replace(r'\sflux\sLast\s\(V\)', '', regex=True)

        # Read the calibration data CSV file
        calibration_data_file_path = 'Heat_flux_sensors_calibration.csv'

        calibration_data = pd.read_csv(calibration_data_file_path, sep=';')
        calibration_data['Sensitivity S0'] = pd.to_numeric(calibration_data['Sensitivity S0'], errors='coerce')
        calibration_data['Correction factor Sc'] = pd.to_numeric(calibration_data['Correction factor Sc'], errors='coerce')

        #T_S=25
        HeatfluxData=pd.DataFrame()
        HeatfluxData['time']=pd.to_datetime(sensor_data['Unnamed: 0'], format='%Y-%m-%dT%H:%M:%S%z')
        if 'CyclerData' in locals():
            CyclerData['DPT Time Adjust'] = CyclerData['DPT Time']
            SensorIndex=(HeatfluxData['time'].dt.tz_localize('UTC+01:00') - CyclerData['DPT Time Adjust'][0]).abs().idxmin()    
        else:
            SensorIndex = 1011

        for column in sensor_data.columns[1:]:
            if column == "E1":
                sensor_id = 'E1'
                T_S = Peltier_control['Current_temp_1'][SensorIndex]
                #print(str(column[0:2]) + ' is ' + T_S + ' °C')
                # print(T_S)
                if np.isnan(T_S):
                    print('Temperature is not a number for ' + column)

                calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
                S_0 = calibration_row['Sensitivity S0']
                S_C = calibration_row['Correction factor Sc']
                S = calculate_sensitivity(S_0, S_C, T_S)

                HeatfluxData['HeatFlux' + column] = calculate_heatflux_vectorized(sensor_data[column], S)
                # #print(column[0:2])
                # calibration_row = calibration_data[calibration_data['serial number'] == sensor_id]
                # HeatfluxData['HeatFlux' + column] = sensor_data.apply(
                # lambda row: calculate_heatflux(row[column], T_S, calibration_row['number'].values[0], calibration_data),
                # axis=1
                # )
            if column == "E2":
                sensor_id = 'E2'
                #T_S=
                T_S = Peltier_control['Current_temp_2'][SensorIndex]
                #print(str(column[0:2]) + ' is ' + T_S + ' °C')
                # print(T_S)
                if np.isnan(T_S):
                    print('Temperature is not a number for ' + column)

                calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
                S_0 = calibration_row['Sensitivity S0']
                S_C = calibration_row['Correction factor Sc']
                S = calculate_sensitivity(S_0, S_C, T_S)

                HeatfluxData['HeatFlux' + column] = calculate_heatflux_vectorized(sensor_data[column], S)
                # #print(column[0:2])
                # calibration_row = calibration_data[calibration_data['serial number'] == sensor_id]
                # HeatfluxData['HeatFlux' + column] = sensor_data.apply(
                # lambda row: calculate_heatflux(row[column], T_S, calibration_row['number'].values[0], calibration_data),
                # axis=1
                # )

HeatfluxData.time_elapsed = (HeatfluxData.time - HeatfluxData.time.iloc[0]).dt.total_seconds()
HeatfluxData.average_heatflux = HeatfluxData.iloc[:, 1:].mean(axis=1)

# try:
#     del metadata_lines, data_lines, lines, line, data, data_section, key, txt_file_path, value
# except:
#     print('')
# del directory, file, filepath, files, prefix, variable_name, calibration_data, calibration_data_file_path, calibration_row, column, T_S, sensor_id, sensor_data_file_path, sensor_data, columns_to_keep, data_columns_Peltier
