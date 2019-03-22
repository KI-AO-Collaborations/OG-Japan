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
import scipy.interpolate as si
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
from ogusa import aggregates as aggr
from ogusa import SS


def chi_n_func(s, a0, a1, a2, a3, a4):
    chi_n = a0 + a1 * s + a2 * s ** 2 + a3 * s ** 3 + a4 * s ** 4
    return chi_n

def chebyshev_func(x, a0, a1, a2, a3, a4):
    func = np.polynomial.chebyshev.chebval(x, [a0, a1, a2, a3, a4])
    return func


def find_moments(p, client):
    b_guess = np.ones((p.S, p.J)) * 0.07
    n_guess = np.ones((p.S, p.J)) * .4 * p.ltilde
    rguess = 0.09
    T_Hguess = 0.12
    factorguess = 7.7 #70000 # Modified
    BQguess = aggr.get_BQ(rguess, b_guess, None, p, 'SS', False)
    exit_early = [0, 10] # 2nd value gives number of valid labor moments to consider before exiting SS_fsolve
    ss_params_baseline = (b_guess, n_guess, None, None, p, client, exit_early)
    guesses = [rguess] + list(BQguess) + [T_Hguess, factorguess]
    [solutions_fsolve, infodict, ier, message] =\
            opt.fsolve(SS.SS_fsolve, guesses, args=ss_params_baseline,
                       xtol=p.mindist_SS, full_output=True)
    rss = solutions_fsolve[0]
    BQss = solutions_fsolve[1:-2]
    T_Hss = solutions_fsolve[-2]
    factor_ss = solutions_fsolve[-1]
    Yss = T_Hss/p.alpha_T[-1]
    fsolve_flag = True
    try:
        output = SS.SS_solver(b_guess, n_guess, rss, BQss, T_Hss,
                        factor_ss, Yss, p, client, fsolve_flag)
    except:
        print('RuntimeError: Steady state aggregate resource constraint not satisfied')
        print('Luckily we caught the error, so minstat_init_calibrate will continue')
        return 1e10

    model_moments = np.array(output['nssmat'].mean(axis=1)[:45]) # calc_moments(output, p.omega_SS, p.lambdas, p.S, p.J)

    return model_moments

def chi_estimate(p, client=None):    
    # Generate labor data moments
    labor_hours = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164])#, 166, 164])
    labor_part_rate = np.array([0.69, 0.849, 0.849, 0.847, 0.847, 0.859, 0.859, 0.709, 0.709])#, 0.212, 0.212])
    employ_rate = np.array([0.937, 0.954, 0.954, 0.966, 0.966, 0.97, 0.97, 0.968, 0.968])#, 0.978, 0.978])
    labor_hours_adj = labor_hours * labor_part_rate * employ_rate
    labor_moments = labor_hours_adj * 12 / (365 * 17.5)
    data_moments_trunc = np.array(list(labor_moments.flatten()))
    ages = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60]) + 2.5
    labor_fun = si.splrep(ages, data_moments_trunc)
    ages_full = np.linspace(21, 65, p.S // 2 + 5)
    data_moments = si.splev(ages_full, labor_fun)

    a0 = 1.25108169e+03
    a1 = -1.19873316e+02
    a2 = 2.20570513e+00
    a3 = -1.76536132e-02
    a4 = 5.19262962e-05

    chi_n = np.ones(p.S)
    chi_n[:p.S // 2 + 5] = chebyshev_func(ages_full, a0, a1, a2, a3, a4)
    slope = chi_n[p.S // 2 + 5 - 1] - chi_n[p.S // 2 + 5 - 2]
    chi_n[p.S // 2 + 5 - 1:] = (np.linspace(65, 100, 36) - 65) * slope + chi_n[p.S // 2 + 5 - 1]
    chi_n[chi_n < 0.5] = 0.5
    p.chi_n = chi_n

    model_moments = find_moments(p, client)
    
    labor_below = np.zeros(p.S // 2 + 5)
    labor_above = np.ones(p.S // 2 + 5) * np.inf
    chi_below = np.zeros(p.S // 2 + 5)
    chi_above = np.zeros(p.S // 2 + 5)

    print('About to start the while loop')

    while ((abs(model_moments - data_moments) > 0.03) & (chi_n[:45] > 0.5)).any()\
        or ((chi_n[:45] <= 0.5) & (labor_above > data_moments)).any():
        both = (labor_below > 0) & (labor_above < np.inf)
        above = (labor_below == 0) & (labor_above < np.inf)
        below = (labor_below > 0) & (labor_above == np.inf)
        chi_n[:45][both] = 0.5 * (chi_below[both] + chi_above[both])
        chi_n[:45][above] = 2 * chi_above[above]
        chi_n[:45][below] = 0.5 * chi_below[below]
        p.chi_n = chi_n
        model_moments = find_moments(p, client)
        above_data_below_above = (model_moments > data_moments) & (model_moments < labor_above)
        below_data_above_below = (model_moments < data_moments) & (model_moments > labor_below)
        labor_above[above_data_below_above] = model_moments[above_data_below_above]
        chi_above[above_data_below_above] = chi_n[:45][above_data_below_above]
        labor_below[below_data_above_below] = model_moments[below_data_above_below]
        chi_below[below_data_above_below] = chi_n[:45][below_data_above_below]
        print('-------------------------------')
        print('New model moments:')
        print(model_moments)
        print('Chi_n:')
        print(chi_n)
        print('-------------------------------')

    print('-------------------------------')
    print('Calibration complete')
    print('Final Chi_n:')
    print(chi_n)
    print('-------------------------------')

    with open("output.txt", "a") as text_file:
        text_file.write('\nFinal model moments: ' + str(model_moments) + '\n')
        text_file.write('\nFinal chi_n: ' + str(chi_n) + '\n')
    pickle.dump(chi_n, open("chi_n.p", "wb"))

    ss_output = SS.run_SS(p)
    return ss_output

    # p.chi_n = chi_n
    # model_moments_best = np.array([0,0,0,0,0,0,0,0,0])
    # while (abs(model_moments - data_moments) > 1e-2).any():
    #     model_moments = find_moments(p)
    #     current_error = model_moments_best - data_moments
    #     new_error = model_moments - data_moments
    #     for i in range(len(new_error)):
    #         if new_error[i] > 1e-2:
    #             if current_error[i] > 0:
    #                 if current_error[i] > new_error[i]:
    #                     p.chi_n[5*i:5(i+1)] 
    #                 else: 
    #                     pass
    #             else: 

def calc_moments(ss_output, omega_SS, lambdas, S, J):
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

    # combine moments
    model_moments = np.array(list(weighted_labor_moments))

    return model_moments
