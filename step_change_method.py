import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 
from datetime import timedelta
import math
import scipy.signal
import os
import pickle
from numpy import exp, sin
from lmfit import minimize, Parameters, fit_report

# code to load data from Sabine's rig and change data format to work with Gavin's code
from setup import *
with open(config_file, 'r') as f:
    inputs = toml.load(f)

L = inputs["L"] # metres, equals 1/2 of cell thickness
deltaT = inputs["deltaT"] # degrees C, magnitude of step change
start_time = inputs["start_time"]+inputs["start_time_addition"] # start time elapsed to fit equation, seconds (the addition is determined manually, from the time diff when power is applied until temperature reaches steady state)
end_time = inputs["end_time"] # end time elapsed to fit equation, seconds
heat_flux_offset = inputs["heat_flux_offset"]
fitting_time_skip = inputs["fitting_time_skip"] # seconds, integer, ignore first few seconds because of overshoot

# getting relevant data points
heat_flux_column = HeatfluxData.average_heatflux + heat_flux_offset
time_window = np.subtract([time for time in HeatfluxData.time_elapsed if start_time <= time <= end_time], start_time)
heat_fluxes = [heat_flux_column[i] for i in range(len(HeatfluxData.time_elapsed)) if start_time <= HeatfluxData.time_elapsed[i] <= end_time]
time_window_for_fitting = [time for time in time_window if time >= fitting_time_skip] # skips first few seconds to ignore overshoots, as defined on top
heat_fluxes_for_fitting = [heat_fluxes[i] for i in range(len(time_window)) if time_window[i] >= fitting_time_skip]




'''
new
'''
def f_series_step_hf(t, alpha, L, k ,dT, num_terms):
    # sums the series coefficients of the given equation
    ans = 0
    for n in range(1, num_terms, 2):    # not the series is for odd integers of n only
        ans += step__hf_sol(t, n, alpha, L, k ,dT)
    return ans

def step__hf_sol(t, n, alpha, L, k, dT):
    s = n*math.pi / (2*L)
    tau = 1 / (s**2*alpha)

    du_dx = (2*dT/L)* np.exp(-t/tau)
    q = du_dx*k
    return q

def r_step(params, x, data):
    alpha = params['alpha']
    L = params['L']
    k = params['k']
    dT = params['dT']
    
    model = f_series_step_hf(x, alpha, L, k, dT, 100)

    return (data - model)

heater_thickness = 18.39e-3    # m
char_length = heater_thickness / 2
rho =  1581.11    # in KJ/kg.K
temp_step = 5   # degC

# best guess as a guide
alpha_guess = 1.8e-07
k_guess = 0.25
x_data_all = fit_data['seconds']
y_guess = f_series_step_hf(x_data_all, alpha_guess, char_length, k_guess, temp_step, 100)
#plt.plot(x_data_all, y_guess, label='Guess')

# fitting to the data
params = Parameters()
params.add('num_terms', value=100, vary=False)
params.add('alpha', value=1.8e-07)
params.add('L', value=char_length, vary=False)
params.add('k', value=0.2)
params.add('dT', value=temp_step, vary=False)   


out = minimize(r_step, params, args=(x_data_trim, y_data_trim))
alpha_fit = out.params['alpha'].value
k_fit = out.params['k'].value
alpha_label = 'Best Fit, alpha: ' + str(np.format_float_scientific(alpha_fit, unique=False, precision=3))

y_fit = f_series_step_hf(x_data_all, alpha_fit, char_length, k_fit, temp_step, 100)
plt.plot(x_data_all, y_fit,'k', label=alpha_label)

cp_fit = k_fit / (alpha_fit * rho)

print('\nFit Alpha: ', alpha_fit)
print('Predicted thermal conductivity: ', k_fit)
print('Predicted specific heat: ', cp_fit)
