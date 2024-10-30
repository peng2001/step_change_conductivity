from numpy.lib.function_base import average
import gw_utils as util
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
