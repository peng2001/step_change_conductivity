import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from setup import *

# Config here
L = 0.00585 # metres, equals 1/2 of cell thickness
deltaT = 5 # degrees C, magnitude of step change
start_time = 1207+10 # start time elapsed to fit equation, seconds (the addition is determined manually, from the time diff when power is applied until temperature reaches steady state)
end_time = 1500 # end time elapsed to fit equation, seconds
heat_flux_offset = 276
fitting_time_skip = 20 # seconds, integer, ignore first few seconds because of overshoot
#################################################################

def step_change_heat_flux(t, conductivity,diffusivityEminus5):
    # t is time since step change in seconds, conductivity and diffusivity are fitting parameters
    # Equation used: qdot = k*infinite series for odd indices((-2*deltaT/L)*exp(-t/tau))
    #   tau = 1/(diffusivity*s^2)
    #   s = n*pi/2L
    summation = 0 # start at zero, add each term in series
    for n in range(1, 100, 2): # loop through odd numbers to 999 to approximate infinite series
        s = n*3.14159265/(2*L)
        tau = 1/((diffusivityEminus5*10**(-5))*(s**2))
        summation += (-2*deltaT/L)*np.exp(-t/tau)
    return conductivity*summation

def fit_heat_flux_equation(time_list, heat_flux_list):
    fit_values, covariance = curve_fit(step_change_heat_flux, xdata=time_list, ydata=heat_flux_list, maxfev=1000000)
    return fit_values

def graph_heat_vs_time_and_fitted_eqn(exp_time, exp_heatflux, conductivity,diffusivity):
    linspace_time = np.arange(exp_time[0]+fitting_time_skip, exp_time[-1], 1)
    fitted_heat_flux = [step_change_heat_flux(t, conductivity, diffusivity) for t in linspace_time]
    plt.plot(exp_time, exp_heatflux, label="Experimental", color="blue")
    plt.plot(linspace_time, fitted_heat_flux, label="Fitted Equation", color="red")
    linspace_time_overshoot = np.arange(2, fitting_time_skip+1, 1)
    fitted_heat_flux_overshoot = [step_change_heat_flux(t, conductivity, diffusivity) for t in linspace_time_overshoot]
    plt.plot(linspace_time_overshoot, fitted_heat_flux_overshoot, color="orange", label="Fitted Equation on Overshoot Area")
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
    print(HeatfluxData.columns)
    print(HeatfluxData.time[start_time])
    heat_flux_column = HeatfluxData.average_heatflux + heat_flux_offset
    time_window = np.subtract([time for time in HeatfluxData.time_elapsed if start_time <= time <= end_time], start_time)
    heat_fluxes = [heat_flux_column[i] for i in range(len(HeatfluxData.time_elapsed)) if start_time <= HeatfluxData.time_elapsed[i] <= end_time]
    time_window_for_fitting = [time for time in time_window if time >= fitting_time_skip] # skips first few seconds to ignore overshoots, as defined on top
    heat_fluxes_for_fitting = [heat_fluxes[i] for i in range(len(time_window)) if time_window[i] >= fitting_time_skip]
    #fitting the analytical solution
    conductivity,diffusivityEminus5 = fit_heat_flux_equation(time_window_for_fitting, heat_fluxes_for_fitting)
    diffusivity = diffusivityEminus5*10**(-5)
    print("**Results**")
    print("Conductivity: "+str(conductivity)+" W/(m*K)")
    print("Diffusivity: "+str(diffusivity)+" m^2/s")
    graph_heat_vs_time(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux)
    graph_heat_vs_time_and_fitted_eqn(time_window, heat_fluxes, conductivity,diffusivityEminus5)