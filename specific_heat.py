import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

from setup import *

# Config here
L = 0.00585 # metres, equals 1/2 of cell thickness
deltaT = 5 # degrees C, magnitude of step change
start_time = 1202 # start time elapsed to fit equation, seconds (the addition is determined manually, from the time diff when power is applied until temperature reaches steady state)
end_time = 2000 # end time elapsed
heat_flux_offset = 275.5
fitting_time_skip = 0 # ignore this for specific heat calculations
#################################################################

def calculate_specific_heat(specific_exp_time, specific_exp_heatflux, num_heat_flux_sensors): # num heat flux sensors is len(HeatfluxData.columns)-1
    sensor_area = 0.04*0.04 # m^2
    mass = 0.897 # kg, cell mass
    mass_per_area = mass/(0.320*0.101)
    total_heat_per_area = -trapezoid(specific_exp_heatflux, specific_exp_time) # losses are already subtracted by the heat flux offset
    # total_heat = num_heat_flux_sensors*total_heat_per_area*sensor_area
    # heat_capacity = total_heat/deltaT
    # specific_heat = heat_capacity/mass
    specific_heat = (total_heat_per_area/deltaT)/mass_per_area
    return specific_heat
    
def graph_heat_vs_time_and_fitted_eqn(specific_exp_time, specific_exp_heatflux):
    plt.plot(specific_exp_time, specific_exp_heatflux, label="Measured Heat Flux", color="blue")
    plt.fill_between(specific_exp_time, specific_exp_heatflux, color='blue', alpha=0.4)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heat Flux (W/m^2)')
    plt.legend()
    plt.title('Heat Flux over Time')
    plt.show()

def graph_heat_vs_time(exp_time, exp_heatflux):
    plt.plot(exp_time, exp_heatflux, marker='o')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heat Flux (W/m^2)')
    plt.title('Heat Flux over Time')
    plt.show()



if __name__ == "__main__":
    # print(HeatfluxData.columns)
    # print(len(HeatfluxData.columns))
    # print(HeatfluxData.time[start_time])
    heat_flux_column = HeatfluxData.average_heatflux + heat_flux_offset
    time_window = np.subtract([time for time in HeatfluxData.time_elapsed if start_time <= time <= end_time], start_time)
    heat_fluxes = [heat_flux_column[i] for i in range(len(HeatfluxData.time_elapsed)) if start_time <= HeatfluxData.time_elapsed[i] <= end_time]
    time_window_for_fitting = [time for time in time_window if time >= fitting_time_skip] # skips first few seconds to ignore overshoots, as defined on top
    heat_fluxes_for_fitting = [heat_fluxes[i] for i in range(len(time_window)) if time_window[i] >= fitting_time_skip]

    graph_heat_vs_time(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux)
    graph_heat_vs_time_and_fitted_eqn(time_window, heat_fluxes)
    specific_heat = calculate_specific_heat(time_window, heat_fluxes, len(HeatfluxData.columns)-1)
    print("\n")
    print("Specific Heat: "+str(specific_heat)+"J/(kg*K)")