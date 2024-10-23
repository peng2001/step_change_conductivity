import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from setup import *

def graph_heat_vs_time(time_elapsed, heatflux):
    plt.plot(time_elapsed, heatflux)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heat Flux (W/m^2)')
    plt.title('Heat Flux over Time')
    plt.show()


if __name__ == "__main__":
    print(HeatfluxData)
    graph_heat_vs_time(HeatfluxData.time_elapsed, HeatfluxData.HeatFluxA0_C05)