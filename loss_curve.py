import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_exponential_curve(initial_value, asymptote_value, time_constant, time_range):
    """
    Fit an exponential curve given initial value, asymptote value, and time constant.
    
    Parameters:
        initial_value (float): The starting value of the curve.
        asymptote_value (float): The value the curve asymptotes towards.
        time_constant (float): The time constant of the curve.
        time_range (tuple): The time range for which to fit the curve (start, end).

    Returns:
        tuple: Fitted parameters (initial_value, asymptote_value, time_constant)
    """
    
    def model(t, initial, asymptote, tau):
        return asymptote + (initial - asymptote) * np.exp(-t / tau)

    # Generate time points
    time_points = np.linspace(time_range[0], time_range[1], 100)

    # Compute the model values using the given parameters
    data_points = model(time_points, initial_value, asymptote_value, time_constant)
    
    # Fit the model to the given data points
    popt, _ = curve_fit(model, time_points, data_points, p0=[initial_value, asymptote_value, time_constant])

    # Extract fitted parameters
    fitted_initial_value, fitted_asymptote_value, fitted_time_constant = popt

    # Print the fitted parameters
    print("Fitted Initial Value:", fitted_initial_value)
    print("Fitted Asymptote Value:", fitted_asymptote_value)
    print("Fitted Time Constant:", fitted_time_constant)

    # Plot the data and the fitted curve
    plt.plot(time_points, data_points, color='green', linestyle='dashed', label='Model Curve')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Exponential Curve Based on Given Parameters')
    plt.show()

    return fitted_initial_value, fitted_asymptote_value, fitted_time_constant

# Example usage
initial_value = 10
asymptote_value = 2
time_constant = 5
time_range = (0, 100)

fit_exponential_curve(initial_value, asymptote_value, time_constant, time_range)
