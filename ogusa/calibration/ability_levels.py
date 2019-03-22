import os
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pickle

# Data: http://www.computer-services.e.u-tokyo.ac.jp/p/cemano/research/DP/documents/coe-f-213.pdf?fbclid=IwAR2Q5JFemo8hNWF-rD7dZshJbz7a7CWFGiPJeUIHwa8iwlYqBmvfgeaZn8Q
ages = np.array([22, 27, 32, 37, 42, 47, 52, 57, 62, 65])
ability = np.array([0.646, 0.843, 0.999, 1.107, 1.165, 1.218, 1.233, 1.127, 0.820, 0.727])
ages = np.array([22, 27, 32, 37, 42, 47, 52, 57, 62, 65])
ability = np.array([0.646, 0.843, 0.999, 1.107, 1.165, 1.218, 1.233, 1.127, 0.820, 0.727])
abil_fun = si.splrep(ages, ability)

def data_moments(vals):
    output = si.splev(vals, abil_fun)
    return output

def model_moments(x, a, b, c, d):
    y = - a * np.arctan(b * x + c) + d 
    return y

def err_vec(params, *args):
    a, b, c, d = params
    vals, = args
    data_mms = data_moments(vals)
    model_mms = model_moments(vals, a, b, c, d)

    sumsq = ((model_mms - data_mms) ** 2).sum()
    return sumsq

def optimize(graph = False, update = False):
    # optimization Problem
    a = 0.5
    b = 0.5
    c = 0.5
    d = 0.5
    params_init = np.array([a,b,c,d])
    gmm_args = np.array([62, 63, 64, 65])

    results_GMM = opt.minimize(err_vec, params_init, args = gmm_args, method = 'L-BFGS-B')
    print(results_GMM)
    a,b,c,d = results_GMM.x

    if graph: 
        # Graphing:
        ages = np.linspace(20, 100, 81)
        ages_full = np.linspace(20, 100, 81)
        ages_beg = np.linspace(20, 65, 46)
        print(ages_beg)
        ages_end = np.linspace(65, 100, 36)
        print(ages_end)
        result_beg = si.splev(ages_beg, abil_fun)
        result_end = model_moments(ages_end, a,b,c,d)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Productivity Level $e_{j,s}$')
        plt.plot(ages_beg, result_beg, color = 'r', label = r'Interpolation')
        plt.legend(loc='upper right')

        plt.plot(ages_end, result_end, color = 'g', label = r'Extrapolation')
        plt.legend(loc='upper right')

        ages_data = np.array([22, 27, 32, 37, 42, 47, 52, 57, 62, 65])
        ability_data = np.array([0.646, 0.843, 0.999, 1.107, 1.165, 1.218, 1.233, 1.127, 0.820, 0.727])
        plt.scatter(ages_data, ability_data, color = 'b', label = r'Literature Values')
        plt.legend(loc='upper right')
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        plt.savefig("ability.png")

    if update:
        #Update Ability Levels in our code: 
        matrix = []
        for i in ages_full:
            line = [4 * i] * 7
            matrix.append(line)
        matrix = pd.DataFrame(matrix)
        print(matrix)
        pickle.dump(matrix, open('run_examples/ability.pkl', 'wb'))

optimize(graph=True)