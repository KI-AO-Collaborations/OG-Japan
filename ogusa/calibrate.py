from __future__ import print_function

'''
This module should be organized as follows:

Main function:
chi_estimate() = returns chi_n, chi_b
    - calls:
        wealth.get_wealth_data() - returns data moments on wealth distribution
        labor.labor_data_moments() - returns data moments on labor supply
        minstat() - returns min of statistical objective function
            model_moments() - returns model moments
                SS.run_SS() - return SS distributions

'''

'''
------------------------------------------------------------------------
Last updated: 7/27/2016

Uses a simulated method of moments to calibrate the chi_n adn chi_b
parameters of OG-USA.

This py-file calls the following other file(s):
    wealth.get_wealth_data()
    labor.labor_data_moments()
    SS.run_SS

This py-file creates the following other file(s): None
------------------------------------------------------------------------
'''

import numpy as np
import scipy.optimize as opt
import pandas as pd
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
from . import wealth
from . import labor
from . import SS
from . import utils


def chi_n_func(s, a0, a1, a2, a3, a4):
    chi_n = a0 + a1 * s + a2 * s ** 2 + a3 * s ** 3 + a4 * s ** 4
    return chi_n

def chebyshev_func(x, a0, a1, a2, a3, a4):
    func = np.polynomial.chebyshev.chebval(x, [a0, a1, a2, a3, a4])
    return func


def chi_estimate(p, client=None):
    '''
    --------------------------------------------------------------------
    This function calls others to obtain the data momements and then
    runs the simulated method of moments estimation by calling the
    minimization routine.

    INPUTS:
    income_tax_parameters = length 4 tuple, (analytical_mtrs, etr_params, mtrx_params, mtry_params)
    ss_parameters         = length 21 tuple, (J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                            g_n_ss, tau_payroll, retire, mean_income_data,\
                            h_wealth, p_wealth, m_wealth, b_ellipse, upsilon)
    iterative_params      = [2,] vector, vector with max iterations and tolerance
                             for SS solution
    chi_guesses           = [J+S,] vector, initial guesses of chi_b and chi_n stacked together
    baseline_dir          = string, path where baseline results located


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    wealth.compute_wealth_moments()
    labor.labor_data_moments()
    minstat()

    OBJECTS CREATED WITHIN FUNCTION:
    wealth_moments     = [J+2,] array, wealth moments from data
    labor_moments      = [S,] array, labor moments from data
    data_moments       = [J+2+S,] array, wealth and labor moments stacked
    bnds               = [S+J,] array, bounds for parameter estimates
    chi_guesses_flat   =  [J+S,] vector, initial guesses of chi_b and chi_n stacked
    min_arg            = length 6 tuple, variables needed for minimizer
    est_output         = dictionary, output from minimizer
    chi_params         = [J+S,] vector, parameters estimates for chi_b and chi_n stacked
    objective_func_min = scalar, minimum of statistical objective function


    OUTPUT:
    ./baseline_dir/Calibration/chi_estimation.pkl


    RETURNS: chi_params
    --------------------------------------------------------------------
    '''

    baseline_dir="./OUTPUT"
    #chi_b_guess = np.ones(80)

    # a0 = 5.38312524e+01
    # a1 = -1.55746248e+00
    # a2 = 1.77689237e-02
    # a3 = -8.04751667e-06
    # a4 = 5.65432019e-08
    """ Kei's Vals
    a0 = 170
    a1 = -2.19154735e+00
    a2 = -2.22817460e-02
    a3 = 4.49993507e-04
    a4 = -1.34197054e-06
    """
    """ Adam's Vals 1
    a0 = 2.59572155e+02
    a1 = -2.35122641e+01
    a2 = 4.27581467e-01
    a3 = -3.40808933e-03
    a4 = 1.00404321e-05
    """

    a0 = 1.10807470e+03#5.19144310e+02
    #a0 = 1.10807470e+03#5.19144310e+02
    a1 = -1.05805189e+02#-4.70245283e+01
    a2 = 1.92411660e+00#8.55162933e-01
    a3 = -1.53364020e-02#-6.81617866e-03
    a4 = 4.51819445e-05#2.00808642e-05
    
 
    # a0 = 2.07381e+02
    # a1 = -1.03143105e+01
    # a2 = 1.42760562e-01
    # a3 = -8.41089078e-04
    # a4 = 1.85173227e-06


    # sixty_plus_chi = 300
    params_init = np.array([a0, a1, a2, a3, a4])

    # Generate labor data moments
    labor_hours = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164])#, 166, 164])

    labor_part_rate = np.array([0.69, 0.849, 0.849, 0.847, 0.847, 0.859, 0.859, 0.709, 0.709])#, 0.212, 0.212])

    employ_rate = np.array([0.937, 0.954, 0.954, 0.966, 0.966, 0.97, 0.97, 0.968, 0.968])#, 0.978, 0.978])

    labor_hours_adj = labor_hours * labor_part_rate * employ_rate

    # get fraction of time endowment worked (assume time
    # endowment is 24 hours minus required time to sleep 6.5 hours)
    labor_moments = labor_hours_adj * 12 / (365 * 17.5)

    #labor_moments[9] = 0.1
    #labor_moments[10] = 0.1

    # combine moments
    data_moments = np.array(list(labor_moments.flatten()))

    # weighting matrix
    W = np.identity(p.J+2+p.S)
    W = np.identity(9)

    ages = np.linspace(20, 65, p.S // 2 + 5)
    #ages = np.linspace(20, 100, p.S)

    est_output = opt.minimize(minstat, params_init,\
                args=(p, client, data_moments, W, ages),\
                method="L-BFGS-B",\
                tol=1e-15, options={'eps': 1e-10})
    a0, a1, a2, a3, a4 = est_output.x
    #chi_n = chebyshev_func(ages, a0, a1, a2, a3, a4)
    chi_n = np.ones(p.S)
    #ages_full = np.linspace(20, 100, p.S)
    #chi_n = chebyshev_func(ages_full, a0, a1, a2, a3, a4)
    
    chi_n[:p.S // 2 + 5] = chebyshev_func(ages, a0, a1, a2, a3, a4)
    slope = 1500#chi_n[p.S // 2 + 5 - 1] - chi_n[p.S // 2 + 5 - 2]
    chi_n[p.S // 2 + 5 - 1:] = (np.linspace(65, 100, 36) - 65) * slope + chi_n[p.S // 2 + 5 - 1]
    
    #chi_n[chi_n < 0.5] = 0.5
    #chi_n[chi_n < 0.5] = 0.5
    p.chi_n = chi_n
    print('PARAMS for Chebyshev:', est_output.x)
    with open("output.txt", "a") as text_file:
        text_file.write('\nPARAMS for Chebyshev: ' + str(est_output.x) + '\n')
    pickle.dump(chi_n, open("chi_n.p", "wb"))

    ss_output = SS.run_SS(p)
    return ss_output


def minstat(params, *args):
    '''
    --------------------------------------------------------------------
    This function generates the weighted sum of squared differences
    between the model and data moments.

    INPUTS:
    chi_guesses = [J+S,] vector, initial guesses of chi_b and chi_n stacked together
    arg         = length 6 tuple, variables needed for minimizer

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    SS.run_SS()
    calc_moments()

    OBJECTS CREATED WITHIN FUNCTION:
    ss_output     = dictionary, variables from SS of model
    model_moments = [J+2+S,] array, moments from the model solution
    distance      = scalar, weighted, squared deviation between data and model moments

    RETURNS: distance
    --------------------------------------------------------------------
    '''

    a0, a1, a2, a3, a4 = params
    p, client, data_moments, W, ages = args
    chi_n = np.ones(p.S)
    #chi_n = chebyshev_func(ages, a0, a1, a2, a3, a4)
    chi_n[:p.S // 2 + 5] = chebyshev_func(ages, a0, a1, a2, a3, a4)
    #chi_n[p.S // 2 + 5:] = sixty_plus_chi
    slope = chi_n[p.S // 2 + 5 - 1] - chi_n[p.S // 2 + 5 - 2]
    chi_n[p.S // 2 + 5 - 1:] = (np.linspace(65, 100, 36) - 65) * slope + chi_n[p.S // 2 + 5 - 1]
    #chi_n[chi_n < 0.5] = 0.5

    p.chi_n = chi_n
    #print(chi_n)

    with open("output.txt", "a") as text_file:
        text_file.write('\nPARAMS AT START\n' + str(params) + '\n')
    print("-----------------------------------------------------")
    print('PARAMS AT START' + str(params))
    print("-----------------------------------------------------")

    try:
       ss_output = SS.run_SS(p, client)
    except:
        with open("output.txt", "a") as text_file:
            text_file.write('\nSteady state not found\n' + str(params) + '\n')
        print("-----------------------------------------------------")
        print("Steady state not found")
        print("-----------------------------------------------------")
        return 1e100

    with open("output.txt", "a") as text_file:
        text_file.write('\nPARAMS AT END\n' + str(params) + '\n')
    print("-----------------------------------------------------")
    print('PARAMS AT END', params)
    print("-----------------------------------------------------")
    model_moments = calc_moments(ss_output, p.omega_SS, p.lambdas, p.S, p.J)
    with open("output.txt", "a") as text_file:
        text_file.write('\nModel moments:\n' + str(model_moments) + '\n')
    print('Model moments:', model_moments)
    print("-----------------------------------------------------")

    # distance with levels
    distance = np.dot(np.dot((np.array(model_moments[:9]) - np.array(data_moments)).T,W),
                   np.array(model_moments[:9]) - np.array(data_moments))
    #distance = ((np.array(model_moments) - np.array(data_moments))**2).sum()
    with open("output.txt", "a") as text_file:
        text_file.write('\nDATA and MODEL DISTANCE: ' + str(distance) + '\n')
    print('DATA and MODEL DISTANCE: ', distance)

    # # distance with percentage diffs
    # distance = (((model_moments - data_moments)/data_moments)**2).sum()

    return distance


def calc_moments(ss_output, omega_SS, lambdas, S, J):
    '''
    --------------------------------------------------------------------
    This function calculates moments from the SS output that correspond
    to the data moments used for estimation.

    INPUTS:
    ss_output = dictionary, variables from SS of model
    omega_SS  = [S,] array, SS population distribution over age
    lambdas   = [J,] array, proportion of population of each ability type
    S         = integer, number of ages
    J         = integer, number of ability types

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    the_inequalizer()

    OBJECTS CREATED WITHIN FUNCTION:
    model_wealth_moments = [J+2,] array, wealth moments from the model
    model_labor_moments  = [S,] array, labor moments from the model
    model_moments        = [J+2+S,] array, wealth and data moments from the model solution
    distance             = scalar, weighted, squared deviation between data and model moments

    RETURNS: distance

    RETURNS: model_moments
    --------------------------------------------------------------------
    '''
    # unpack relevant SS variables
    n = ss_output['nssmat']

    # labor moments
    model_labor_moments = (n.reshape(S, J) * lambdas.reshape(1, J)).sum(axis=1)

    ### we have ages 20-100 so lets find binds based on population weights
    # converting to match our data moments
    model_labor_moments = pd.DataFrame(model_labor_moments * omega_SS)
    model_labor_moments.rename({0: 'labor_weighted'}, axis=1, inplace=True)

    ages = np.linspace(20, 100, S)
    age_bins = np.linspace(20, 75, 12)
    age_bins[11] = 101
    labels = np.linspace(20, 70, 11)
    model_labor_moments['pop_dist'] = omega_SS
    model_labor_moments['age_bins'] = pd.cut(ages, age_bins, right=False, include_lowest=True, labels=labels)
    weighted_labor_moments = model_labor_moments.groupby('age_bins')['labor_weighted'].sum() /\
                                model_labor_moments.groupby('age_bins')['pop_dist'].sum()


    # For visualization purpose:



    # combine moments
    model_moments = list(weighted_labor_moments)

    return model_moments
