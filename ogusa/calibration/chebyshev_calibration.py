#%%
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

def func(x, a0, a1, a2, a3, a4):
    func = np.polynomial.chebyshev.chebval(x, [a0, a1, a2, a3, a4])
    return func

def get_data_moments():
    # Labor Data Moments
    labor_hours = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164, 166, 164])
    labor_part_rate = np.array([0.69, 0.849, 0.849, 0.847, 0.847, 0.859, 0.859, 0.709, 0.709, 0.212, 0.212])
    employ_rate = np.array([0.937, 0.954, 0.954, 0.966, 0.966, 0.97, 0.97, 0.968, 0.968, 0.978, 0.978])
    labor_hours_adj = labor_hours * labor_part_rate * employ_rate
        # get fraction of time endowment worked (assume time
        # endowment is 24 hours minus required time to sleep 6.5 hours)
    data_moments = labor_hours_adj * 12 / (365 * 17.5)
    return data_moments

#%%
### Laboratory:
def get_values(a0,a1,a2,a3,a4):
    x = np.linspace(20, 65, 46)
    return func(x, a0,a1,a2,a3,a4)

def get_chebyshev_params(data_moments):
    '''
    #data_moments = np.array([38.12000874, 33.22762421, 25.3484224, 26.67954008, 24.41097278, 23.15059004, 22.46771332, 21.85495452, 21.46242013, 22.00364263, 21.57322063, 21.53371545, 21.29828515, 21.10144524, 20.8617942, 20.57282, 20.47473172, 20.31111347, 19.04137299, 18.92616951, 20.58517969, 20.48761429, 20.21744847, 19.9577682, 19.66931057, 19.6878927, 19.63107201, 19.63390543, 19.5901486, 19.58143606, 19.58005578, 19.59073213, 19.60190899, 19.60001831, 21.67763741, 21.70451784, 21.85430468, 21.97291208, 21.97017228, 22.25518398, 22.43969757, 23.21870602, 24.18334822, 24.97772026, 26.37663164, 29.65075992, 30.46944758, 31.51634777, 33.13353793, 32.89186997, 38.07083882, 39.2992811, 40.07987878, 35.19951571, 35.97943562, 37.05601334, 37.42979341, 37.91576867, 38.62775142, 39.4885405, 37.10609921, 40.03988031, 40.86564363, 41.73645892, 42.6208256, 43.37786072, 45.38166073, 46.22395387, 50.21419653, 51.05246704, 53.86896121, 53.90029708, 61.83586775, 64.87563699, 66.91207845, 68.07449767, 71.27919965, 73.57195873, 74.95045988, 76.6230815])
    '''
    ages = np.linspace(20, 100, 81)
    return np.polynomial.chebyshev.chebfit(ages, data_moments, 4)

a0 = 1.10807470e+03
a1 = -1.05805189e+02
a2 = 1.92411660e+00
a3 = -1.53364020e-02
a4 = 4.51819445e-05
get_values(a0,a1,a2,a3,a4)


#%%
# Chi_n Graph
def optimize(chi_graph = True, moment_graph = True):
    # Parameter Guesses 
    a0 = 1.10807470e+03#5.19144310e+02
    a1 = -1.05805189e+02#-4.70245283e+01
    a2 = 1.92411660e+00#8.55162933e-01
    a3 = -1.53364020e-02#-6.81617866e-03
    a4 = 4.51819445e-05#

    if chi_graph:
    # Plot of Chi_n
        ages_beg = np.linspace(20, 65, 46)
        data_beg = func(ages_beg, a0, a1,a2,a3,a4)
        ages_end = np.linspace(65, 100, 36)
        data_end = (data_beg[-1] - data_beg[-2]) * (ages_end - 65) + data_beg[-1]
        data = np.linspace(20, 100, 81)
        ages = np.linspace(20, 100, 81)
        data[:46] = data_beg
        data[45:] = data_end
        plt.xlabel('Age $s$')
        plt.ylabel(r'$\chi_n^s$')
        plt.plot(ages, data, color = 'r', label = r'Estimated')
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        plt.savefig("chi_n.png")
        plt.close()

    # Labor model_moments from Simulation:
    model_moments = np.array([0.2040540765626132, 0.2526354466647778, 0.2769057405403771, 0.27696186745847984, 0.27235828933517375, 0.27723433376871154, 0.2881355407467818, 0.2541506887816195, 0.1789017426314303, 0.11635354318881265, 0.059785018218161144])
    model_moments = np.array([0.1499162057778458, 0.22552815020838995, 0.26996153865471983, 0.2691089631249235, 0.25935860093557483, 0.2560840007407618, 0.2423648503430985, 0.18459865014459426, 0.1062297896818786, 0.0058177805287718135, 0.0011070846641351186])
    labels = np.linspace(20, 70, 11)
    labels[-1] = 85

    if moment_graph:
        # labor data moments:
        data_moments = get_data_moments()
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Labor Supply $\frac{\bar{n_s}}{\tilde{l}}$')
        plt.scatter(labels, data_moments, color = 'r', label = r'Data Moments')
        plt.legend(loc='upper right')
        plt.scatter(labels, model_moments, color = 'b', label = r'Model Moments')
        plt.legend(loc='upper right')
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        plt.savefig("labor_moments.png")
        plt.close()


optimize()