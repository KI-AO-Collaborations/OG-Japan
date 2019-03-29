'''
AUTHORS: Kei Irizawa and Adam Oppenheimer
DATE: March 22, 2019
COPYRIGHT MARCH 22, 2019 KEI IRIZAWA AND ADAM OPPENHEIMER
'''


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
    rguess = 0.08961277823002804 # 0.09
    T_Hguess = 0.12
    factorguess = 12.73047710050195 # 7.7 #70000 # Modified
    BQguess = aggr.get_BQ(rguess, b_guess, None, p, 'SS', False)
    exit_early = [0, -1] # 2nd value gives number of valid labor moments to consider before exiting SS_fsolve
                         # Put -1 to run to SS
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
    '''
    COPYRIGHT MARCH 22, 2019 KEI IRIZAWA AND ADAM OPPENHEIMER
    '''
    # Generate labor data moments
    labor_hours = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164])#, 166, 164])
    labor_part_rate = np.array([0.69, 0.849, 0.849, 0.847, 0.847, 0.859, 0.859, 0.709, 0.709])#, 0.212, 0.212])
    employ_rate = np.array([0.937, 0.954, 0.954, 0.966, 0.966, 0.97, 0.97, 0.968, 0.968])#, 0.978, 0.978])
    labor_hours_adj = labor_hours * labor_part_rate * employ_rate
    labor_moments = labor_hours_adj * 12 / (365 * 17.5)
    data_moments_trunc = np.array(list(labor_moments.flatten()))
    #ages = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60]) + 2.5
    #labor_fun = si.splrep(ages, data_moments_trunc)
    #ages_full = np.linspace(21, 65, p.S // 2 + 5)
    #data_moments = si.splev(ages_full, labor_fun)
    data_moments = np.repeat(data_moments_trunc, 5) # Set labor values to equal average over bin

    # a0 = 1.25108169e+03
    # a1 = -1.19873316e+02
    # a2 = 2.20570513e+00
    # a3 = -1.76536132e-02
    # a4 = 5.19262962e-05

    # chi_n = np.ones(p.S)
    # chi_n[:p.S // 2 + 5] = chebyshev_func(ages_full, a0, a1, a2, a3, a4)
    # slope = chi_n[p.S // 2 + 5 - 1] - chi_n[p.S // 2 + 5 - 2]
    # chi_n[p.S // 2 + 5 - 1:] = (np.linspace(65, 100, 36) - 65) * slope + chi_n[p.S // 2 + 5 - 1]
    # chi_n[chi_n < 0.5] = 0.5
    chi_n = pickle.load(open("chi_n.p", "rb"))
    p.chi_n = chi_n

    model_moments = find_moments(p, client)
    
    labor_below = np.zeros(p.S // 2 + 5)
    labor_above = np.ones(p.S // 2 + 5) * np.inf
    chi_below = np.zeros(p.S // 2 + 5)
    chi_above = np.zeros(p.S // 2 + 5)

    chi_prev = np.zeros(p.S // 2 + 5)
    consec_above = np.zeros(p.S // 2 + 5)
    consec_below = np.zeros(p.S // 2 + 5)

    print('About to start the while loop')
    eps_val = 0.001
    #still_calibrate = ((abs(model_moments - data_moments) > eps_val) & (chi_n[:45] > 0.5))\
    #    | ((chi_n[:45] <= 0.5) & (labor_above > data_moments))
    still_calibrate = (abs(model_moments - data_moments) > eps_val) & ((chi_n[:45] > 0.5)\
        | ((chi_n[:45] <= 0.5) & (labor_above > data_moments)))
    moments_calibrated_per_step = []

    while still_calibrate.any():
        ### Check that 2 consecutive chi_n estimates aren't equal
        if (chi_n[:45] == chi_prev).all():
            raise RuntimeError('Calibration failure. No chi_n values changed between guesses')
        chi_prev = np.copy(chi_n[:45])
        ### Set above/below arrays based on model moments
        above_data_below_above = (model_moments > data_moments) #& (model_moments < labor_above)
        below_data_above_below = (model_moments < data_moments) #& (model_moments > labor_below)
        # Had to comment out checking if closer than previous guess because if
        # the result moves, the convex combination might be outside the range
        # and it gets stuck in an infinite loop because the guess never improves
        labor_above[above_data_below_above] = model_moments[above_data_below_above]
        chi_above[above_data_below_above] = chi_n[:45][above_data_below_above]
        labor_below[below_data_above_below] = model_moments[below_data_above_below]
        chi_below[below_data_above_below] = chi_n[:45][below_data_above_below]
        ### Set consecutive above/below values
        consec_above[above_data_below_above] += 1
        consec_above[below_data_above_below] = 0
        consec_below[below_data_above_below] += 1
        consec_below[above_data_below_above] = 0
        consec = (consec_above >= 4) | (consec_below >= 4)
        ### Create arrays for labor boundaries
        print(str(np.sum(still_calibrate)) + ' labor moments are still being calibrated')
        moments_calibrated_per_step.append(np.sum(still_calibrate))
        print('Moments calibrated at each iteration (including this iteration):')
        print(moments_calibrated_per_step)
        both = (((labor_below > 0) & (labor_above < np.inf)) |\
            ((labor_below == 0) & (labor_above == np.inf))) & (still_calibrate)
        above = (labor_below == 0) & (labor_above < np.inf) & (still_calibrate)
        below = (labor_below > 0) & (labor_above == np.inf) & (still_calibrate)
        print(str(np.sum(both)) + ' labor moments are being convexly shifted')
        print(str(np.sum(above)) + ' labor moments are being shifted down')
        print(str(np.sum(below)) + ' labor moments are being shifted up')
        ### Calculate convex combination factor
        above_dist = abs(labor_above - data_moments)
        below_dist = abs(data_moments - labor_below)
        total_dist = above_dist + below_dist
        above_factor = below_dist / total_dist
        below_factor = above_dist / total_dist
        #### Adjust by convex combination factor
        chi_n[:45][both] = np.copy(below_factor[both] * chi_below[both] +\
            above_factor[both] * chi_above[both])
        invalid_factor = np.isnan(chi_n[:45][both]) # Modified
        chi_n[:45][both][invalid_factor] = np.copy(0.5 * (chi_below[both][invalid_factor] + chi_above[both][invalid_factor])) # Modified
        ### Adjust values that aren't bounded both above and below by labor error factors
        error_factor = model_moments / data_moments
        chi_n[:45][above] = np.copy(np.minimum(error_factor[above] * chi_above[above], 1.02 * chi_above[above]))#np.copy(1.02 * chi_above[above])
        chi_n[:45][below] = np.copy(np.maximum(error_factor[below] * chi_below[below], 0.98 * chi_below[below]))#np.copy(0.98 * chi_below[below])
        ### Solve moments using new chi_n guesses
        p.chi_n = chi_n
        model_moments = find_moments(p, client)
        print('-------------------------------')
        print('New model moments:')
        print(list(model_moments))
        print('Chi_n:')
        print(list(chi_n))
        print('-------------------------------')
        print('Labor moment differences:')
        print(model_moments[still_calibrate] - data_moments[still_calibrate])
        print('-------------------------------')
        ### Redefine still_calibrate and both based on new model moments
        still_calibrate = (abs(model_moments - data_moments) > eps_val) & ((chi_n[:45] > 0.5)\
            | ((chi_n[:45] <= 0.5) & (labor_above > data_moments)))
        both = (((labor_below > 0) & (labor_above < np.inf)) |\
            ((labor_below == 0) & (labor_above == np.inf))) & (still_calibrate)
        print('Chi differences:')
        print(chi_below[still_calibrate] - chi_above[still_calibrate])
        print('-------------------------------')
        print('Chi below:')
        print(chi_below[still_calibrate])
        print('-------------------------------')
        print('Chi above:')
        print(chi_above[still_calibrate])
        print('-------------------------------')
        print('Labor above:')
        print(labor_above[still_calibrate])
        print('-------------------------------')
        print('Labor below:')
        print(labor_below[still_calibrate])
        print('-------------------------------')
        ### Fix stuck boundaries
        #still_calibrate_stuck_1 = ((abs(model_moments - data_moments) > eps_val) & (chi_n[:45] > 0.5))\
        #| ((chi_n[:45] <= 0.5) & (labor_above > data_moments))
        #still_calibrate_stuck_2 = ((abs(model_moments - data_moments) > 10 * eps_val) & (chi_n[:45] > 0.5))\
        #| ((chi_n[:45] <= 0.5) & (labor_above > data_moments))
        #stuck_1 = ((chi_below - chi_above) < 10 * eps_val) & (still_calibrate_stuck_1)
        #stuck_2 = ((chi_below - chi_above) < 1e3 * eps_val) & (still_calibrate_stuck_2)
        #stuck = (stuck_1) | (stuck_2)
        stuck = ((chi_below - chi_above) < 10) & (consec) & (both)
        if (stuck).any():
            consec_above[stuck] = 0
            consec_below[stuck] = 0
            check_above_stuck = (stuck) & (model_moments > data_moments)
            check_below_stuck = (stuck) & (model_moments < data_moments)
            print(str(np.sum(check_above_stuck)) + ' labor moments are being checked to see if they are too high')
            print(str(np.sum(check_below_stuck)) + ' labor moments are being checked to see if they are too low')
            ### Make sure chi_n bounds are still valid
            check_chi_n = chi_n.copy()
            check_chi_n[:45][check_above_stuck] = np.copy(chi_below[check_above_stuck])
            check_chi_n[:45][check_below_stuck] = np.copy(chi_above[check_below_stuck])
            p.chi_n = check_chi_n
            check_model_moments = find_moments(p, client)
            above_stuck = (check_above_stuck) & (check_model_moments > data_moments)
            below_stuck = (check_below_stuck) & (check_model_moments < data_moments)
            print(str(np.sum(above_stuck)) + ' labor moments are being unstuck from being too high')
            print(str(np.sum(below_stuck)) + ' labor moments are being unstuck from being too low')
            total_stuck = str(np.sum(above_stuck) + np.sum(below_stuck))
            moments_calibrated_per_step.append(str(np.sum(stuck)) + '(checked) ' + total_stuck + '(stuck)')
            print('Moments calibrated at each iteration (including this iteration):')
            print(moments_calibrated_per_step)
            labor_below[above_stuck] = 0
            labor_above[below_stuck] = np.inf
            chi_below[above_stuck] *= 2
            chi_above[below_stuck] *= 0.5
            still_calibrate = (abs(check_model_moments - data_moments) > eps_val) & ((check_chi_n[:45] > 0.5)\
                | ((check_chi_n[:45] <= 0.5) & (labor_above > data_moments)))
            if not (still_calibrate).any():
                chi_n = check_chi_n
                model_moments = check_model_moments
        else:
            print('No labor moments are stuck')
        

    print('-------------------------------')
    print('Calibration complete')
    print('Final Chi_n:')
    print(list(chi_n))
    print('-------------------------------')
    print('Final model moments:')
    print(list(model_moments))
    print('-------------------------------')
    print('Moments calibrated at each iteration:')
    print(moments_calibrated_per_step)
    print('Number of iterations to solve:')
    print(len(moments_calibrated_per_step))
    print('-------------------------------')

    with open("output.txt", "a") as text_file:
        text_file.write('\nFinal model moments: ' + str(model_moments) + '\n')
        text_file.write('\nFinal chi_n: ' + str(chi_n) + '\n')
    pickle.dump(chi_n, open("chi_n.p", "wb"))
    stop
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
