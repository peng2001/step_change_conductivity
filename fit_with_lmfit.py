import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
from setup import *
import math

##########################################################

with open(config_file, 'r') as f:
    inputs = toml.load(f)

L_orig = inputs["L"] # metres, equals 1/2 of cell thickness
deltaT_orig = inputs["deltaT"] # degrees C, magnitude of step change
HeatfluxData_orig = HeatfluxData
start_time = inputs["start_time"]+inputs["start_time_addition"] # start time elapsed to fit equation, seconds (the addition is determined manually, from the time diff when power is applied until temperature reaches steady state)
end_time = inputs["end_time"] # end time elapsed to fit equation, seconds
fitting_time_skip = inputs["fitting_time_skip"] # seconds, integer, ignore first few seconds because of overshoot

def step_change_heat_flux(t, conductivity,diffusivityEminus5,heat_flux_offset):
    # t is time since step change in seconds, conductivity and diffusivity are fitting parameters
    # Equation used: qdot = k*infinite series for odd indices((-2*deltaT/L)*exp(-t/tau))
    #   tau = 1/(diffusivity*s^2)
    #   s = n*pi/2L
    summation = 0 # start at zero, add each term in series
    for n in range(1, 6, 2): # loop through odd numbers to 100 to approximate infinite series
        s = n*3.14159265/(2*L)
        tau = 1/((diffusivityEminus5*10**(-5))*(s**2))
        summation += (-2*deltaT/L)*np.exp(-t/tau)
    return conductivity*summation + heat_flux_offset

def exponential(t, initial, asymptote, tau):
    return asymptote + (initial - asymptote) * np.exp(-t / tau)

def fit_exponential(time_list, heat_flux_list):
    initial_value_guess = 0
    asymptote_guess = HeatfluxData.average_heatflux[next(i for i, t in enumerate(HeatfluxData.time_elapsed) if t >= inputs["end_time"])]
    tau_guess = 3000
    popt, pcov = curve_fit(exponential, time_list, heat_flux_list, p0=[initial_value_guess, asymptote_guess, tau_guess])
    fitted_initial, fitted_asymptote, fitted_tau = popt
    return fitted_initial, fitted_asymptote, fitted_tau

def fit_heat_flux_equation(time_list, heat_flux_list):
    model = Model(step_change_heat_flux)
    k_guess = -20
    alpha_guess = 1.5
    offset_guess = 0
    params = model.make_params(conductivity=k_guess,diffusivityEminus5=alpha_guess,heat_flux_offset=offset_guess)
    params['heat_flux_offset'].set(value=offset_guess, vary=False) # FIX IT SO THAT IT WONT BE FITTED
    result = model.fit(heat_flux_list, params, t=time_list)
    return result

def round_sig(x, sig):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

def graph_heat_vs_time_and_fitted_eqn(exp_time, exp_heatflux, adjusted_heat_flux, conductivity, diffusivity, heat_flux_offset):
    linspace_time = np.arange(exp_time[0]+fitting_time_skip, exp_time[-1], 1)
    fitted_heat_flux = [step_change_heat_flux(t, conductivity, diffusivity, heat_flux_offset) for t in linspace_time]
    plt.plot(exp_time, exp_heatflux, label="Experimental", color="blue")
    plt.plot(exp_time, adjusted_heat_flux, label="Experimental with Losses Removed", color="purple")
    plt.plot(linspace_time, fitted_heat_flux, label="Fitted Equation", color="red")
    linspace_time_overshoot = np.arange(2, fitting_time_skip+1, 1)
    fitted_heat_flux_overshoot = [step_change_heat_flux(t, conductivity, diffusivity, heat_flux_offset) for t in linspace_time_overshoot]
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

def calculate_fit_error(exp_time, exp_heatflux, conductivity,diffusivity,heat_flux_offset):
    linspace_time = np.arange(exp_time[0]+fitting_time_skip, exp_time[-1], 0.1)
    fitted_heat_flux = [step_change_heat_flux(t, conductivity, diffusivity, heat_flux_offset) for t in linspace_time]
    # filter values in experimental value to only include points that are used to fit equation
    filtered_exp_time = [time for time in exp_time if linspace_time[0] <= time <= linspace_time[-1]]
    filtered_exp_heat = [exp_heatflux[i] for i in range(len(exp_time)) if linspace_time[0] <= exp_time[i] <= linspace_time[-1]]
    interpolated_fitted_heat_flux = np.interp(filtered_exp_time, linspace_time, fitted_heat_flux)
    avg_abs_relative_err = np.sum(np.abs(np.subtract(filtered_exp_heat,interpolated_fitted_heat_flux)))/np.sum(np.abs(filtered_exp_heat))
    print("Average absolute relative error of heat equation flux fit: "+str(100*avg_abs_relative_err)+" %")

def run_fitting():
    heat_flux_column = HeatfluxData.average_heatflux
    time_window = np.subtract([time for time in HeatfluxData.time_elapsed if start_time <= time <= end_time], start_time)
    heat_fluxes = [heat_flux_column[i] for i in range(len(HeatfluxData.time_elapsed)) if start_time <= HeatfluxData.time_elapsed[i] <= end_time]
    time_window_for_fitting = [time for time in time_window if time >= fitting_time_skip] # skips first few seconds to ignore overshoots, as defined on top
    heat_fluxes_for_fitting = [heat_fluxes[i] for i in range(len(time_window)) if time_window[i] >= fitting_time_skip]
    #fitting the analytical solution
    # result_direct = fit_heat_flux_equation(time_window_for_fitting, heat_fluxes_for_fitting)
    # heat_flux_offset = result_direct.params['heat_flux_offset'].value
    # conductivity = result_direct.params['conductivity'].value
    # conductivity_error = result_direct.params['conductivity'].stderr
    # diffusivityEminus5 = result_direct.params['diffusivityEminus5'].value
    # diffusivityEminus5_error = result_direct.params['diffusivityEminus5'].stderr
    # diffusivity = (diffusivityEminus5)*10**(-5)
    # diffusivity_error = (diffusivityEminus5_error)*10**(-5)
    # print("**Results**")
    # print("Conductivity: "+str(round_4_sig(conductivity))+" W/(m*K)")
    # print("Diffusivity: "+str(round_4_sig(diffusivity))+" m^2/s")
    # print("Conductivity stderr: "+str(conductivity_error)+" W/(m*K)")
    # print("Diffusivity stderr: "+str(diffusivity_error)+" m^2/s")
    # print("Heat flux offset: "+str(round_4_sig(heat_flux_offset))+" W/m^2")
    # graph_heat_vs_time(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux)
    # graph_heat_vs_time_and_fitted_eqn(time_window, heat_fluxes, conductivity,diffusivityEminus5,heat_flux_offset)

    initial_loss_index = next(i for i, t in enumerate(HeatfluxData.time_elapsed) if t >= inputs["start_time"]) - 5 # 5 points before the start time to get initial loss
    prev_50_points = HeatfluxData.average_heatflux[(initial_loss_index-50):initial_loss_index] # get prev 100 points to get the average initial loss
    initial_loss_estimate = sum(prev_50_points) / len(prev_50_points)
    # print("Initial loss: "+str(initial_loss_estimate))

    fitted_initial, fitted_asymptote, fitted_tau = fit_exponential(time_window_for_fitting, heat_fluxes_for_fitting)
    # print("Final loss: "+str(fitted_asymptote))
    # print("HEat flux fitted to exponential:")
    # print(fitted_initial)
    # print(fitted_asymptote)
    # print(fitted_tau)
    losses = [exponential(t, initial=initial_loss_estimate, asymptote=fitted_asymptote, tau=fitted_tau) for t in time_window]
    linspace_time = np.arange(time_window[0]+fitting_time_skip, time_window[-1], 1)
    adjusted_heat_flux = np.subtract(heat_fluxes, losses)
    adjusted_heat_fluxes_for_fitting = [adjusted_heat_flux[i] for i in range(len(time_window)) if time_window[i] >= fitting_time_skip]
    fitted_exponential = [exponential(t, fitted_initial, fitted_asymptote, fitted_tau) for t in linspace_time]
    # plt.plot(time_window, heat_fluxes, label="Experimental", color="blue")
    # # plt.plot(time_window, loss_curve, label="Losses", color="green")
    # plt.plot(time_window, adjusted_heat_flux, label="Heat flux with losses adjusted", color="purple")
    # linspace_time_overshoot = np.arange(2, fitting_time_skip+1, 1)
    # plt.plot(linspace_time, fitted_exponential, color="black", label="Fitted Exponential")
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Heat Flux (W/m^2)')
    # plt.legend()
    # plt.title('Heat Flux over Time')
    # plt.show()

    result = fit_heat_flux_equation(time_window_for_fitting, adjusted_heat_fluxes_for_fitting)
    # heat_flux_offset = result.params['heat_flux_offset'].value
    conductivity = result.params['conductivity'].value
    conductivity_error = result.params['conductivity'].stderr
    diffusivityEminus5 = result.params['diffusivityEminus5'].value
    diffusivityEminus5_error = result.params['diffusivityEminus5'].stderr
    diffusivity = (diffusivityEminus5)*10**(-5)
    diffusivity_error = (diffusivityEminus5_error)*10**(-5)
    # print("Heat flux offset: "+str(round_4_sig(heat_flux_offset))+" W/m^2")
    # graph_heat_vs_time(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux)
    # graph_heat_vs_time_and_fitted_eqn(time_window, heat_fluxes, adjusted_heat_flux, conductivity,diffusivityEminus5,heat_flux_offset=0)
    return conductivity, diffusivity, conductivity_error, diffusivity_error



if __name__ == "__main__":
    print("Fitting using measured parameters")
    L = L_orig; deltaT = deltaT_orig; HeatfluxData = HeatfluxData_orig
    conductivity, diffusivity, conductivity_error, diffusivity_error = run_fitting()
    print("**Results**")
    # print("Conductivity: "+str(round_4_sig(conductivity))+" W/(m*K)")
    # print("Diffusivity: "+str(round_4_sig(diffusivity))+" m^2/s")
    # print("Conductivity stderr: "+str(conductivity_error)+" W/(m*K)")
    # print("Diffusivity stderr: "+str(diffusivity_error)+" m^2/s")

    delta_ks = []
    delta_alphas = []

    # print("______________________________")
    # print("Fitting using L + uncertainty")
    L = L_orig+0.001
    conductivity_new, diffusivity_new, conductivity_error_new, diffusivity_error_new = run_fitting()
    delta_k_1 = conductivity_new-conductivity
    delta_alpha_1 = diffusivity_new-diffusivity
    # print("delta k: "+str(conductivity_new-conductivity))
    # print("delta alpha: "+str(diffusivity_new-diffusivity))
    # print("Fitting using L - uncertainty")
    L = L_orig-0.001
    conductivity_new, diffusivity_new, conductivity_error_new, diffusivity_error_new = run_fitting()
    delta_k_2 = conductivity_new-conductivity
    delta_alpha_2 = diffusivity_new-diffusivity
    # print("delta k: "+str(conductivity_new-conductivity))
    # print("delta alpha: "+str(diffusivity_new-diffusivity))
    delta_ks.append(abs(max(abs(delta_k_1), abs(delta_k_2))))
    delta_alphas.append(abs(max(abs(delta_alpha_1), abs(delta_alpha_2))))

    # print("______________________________")
    # print("Fitting using deltaT + uncertainty")
    deltaT = deltaT_orig+0.01
    conductivity_new, diffusivity_new, conductivity_error_new, diffusivity_error_new = run_fitting()
    delta_k_1 = conductivity_new-conductivity
    delta_alpha_1 = diffusivity_new-diffusivity
    # print("delta k: "+str(conductivity_new-conductivity))
    # print("delta alpha: "+str(diffusivity_new-diffusivity))
    # print("Fitting using deltaT - uncertainty")
    deltaT = deltaT_orig-0.01
    conductivity_new, diffusivity_new, conductivity_error_new, diffusivity_error_new = run_fitting()
    delta_k_2 = conductivity_new-conductivity
    # delta_alpha_2 = diffusivity_new-diffusivity
    # print("delta k: "+str(conductivity_new-conductivity))
    # print("delta alpha: "+str(diffusivity_new-diffusivity))
    delta_ks.append(abs(max(abs(delta_k_1), abs(delta_k_2))))
    delta_alphas.append(abs(max(abs(delta_alpha_1), abs(delta_alpha_2))))

    # print("______________________________")
    # print("Fitting using Heat flux + uncertainty and start, - uncertainty at end")
    HeatfluxData.average_heatflux = np.multiply(HeatfluxData_orig.average_heatflux, np.linspace(1.03, 0.97, len(HeatfluxData_orig.average_heatflux)))
    conductivity_new, diffusivity_new, conductivity_error_new, diffusivity_error_new = run_fitting()
    delta_k_1 = conductivity_new-conductivity
    delta_alpha_1 = diffusivity_new-diffusivity
    # print("delta k: "+str(conductivity_new-conductivity))
    # print("delta alpha: "+str(diffusivity_new-diffusivity))
    # print("Fitting using Heat flux - uncertainty and start, + uncertainty at end")
    HeatfluxData.average_heatflux = np.multiply(HeatfluxData_orig.average_heatflux, np.linspace(0.97, 1.03, len(HeatfluxData_orig.average_heatflux)))
    conductivity_new, diffusivity_new, conductivity_error_new, diffusivity_error_new = run_fitting()
    delta_k_2 = conductivity_new-conductivity
    delta_alpha_2 = diffusivity_new-diffusivity
    # print("delta k: "+str(conductivity_new-conductivity))
    # print("delta alpha: "+str(diffusivity_new-diffusivity))
    delta_ks.append(abs(max(abs(delta_k_1), abs(delta_k_2))))
    delta_alphas.append(abs(max(abs(delta_alpha_1), abs(delta_alpha_2))))
    
    sum_of_errors_k = np.sum(delta_ks+conductivity_error)
    sum_of_errors_alpha = np.sum(delta_alphas+diffusivity_error)
    print("Conductivity: "+str(round_sig(conductivity, 4))+" W/(m*K)"+" +- "+str(round_sig((sum_of_errors_k/conductivity)*100, 2))+"%")
    print("Diffusivity: "+str(round_sig(diffusivity, 4))+" m^2/s"+" +-" +str(round_sig((sum_of_errors_alpha/diffusivity)*100, 2))+"%")
    