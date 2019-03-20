#%%
'''
------------------------------------------------------------------------
Functions for generating demographic objects necessary for the OG-USA
model

This module defines the following function(s):
    get_fert()
    get_mort()
    pop_rebin()
    get_imm_resid()
    immsolve()
    get_pop_objs()
------------------------------------------------------------------------
'''
#%%
# Import packages
import os
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
#%%
'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''
#%%
def select_fert_data(fert, set_zeroes=False):
        new_fert = fert[fert["AgeDef"] == "  ARDY"]
        new_fert = new_fert[new_fert["Collection"] == "       HFD"]
        new_fert = new_fert[(new_fert["   RefCode"] == "    JPN_11")]
        new_fert.drop(["AgeDef", "Collection", "   RefCode"], axis=1, inplace=True)
        new_fert.columns = ["Year", "Age", "Values"]
        if set_zeroes:
                new_fert["Values"][new_fert["Age"] == 14] = 0
                new_fert["Values"][new_fert["Age"] == 15] = 0
                new_fert["Values"][new_fert["Age"] == 49] = 0
                new_fert["Values"][new_fert["Age"] == 50] = 0
        return new_fert.astype(float)
#%%
def get_fert(totpers, min_yr, max_yr, graph=False):
    pop_filename = "/data/Population.csv"
    pop_data = pd.read_csv(pop_filename, sep='\s+', usecols=["Year", "Age", "Total"])
    pop_data = select_data(pop_data)
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    curr_pop = np.array(pop_data_samp[pop_data_samp['Year'] == 2014]["Total"], dtype='f')
    curr_pop_pct = curr_pop / curr_pop.sum() # pct population of that age group within same year
    # Get fertility rate by age-bin data

    fert_filename = "/data/Fertility.csv"
    fert_data = pd.read_csv(fert_filename,\
        usecols=["Year1", "Age", "   ASFR", "AgeDef",\
                        "Collection", "   RefCode"])
    fert_data = select_fert_data(fert_data)
    fert_list = []
    for i in range(14, 51):
        age = fert_data[fert_data["Age"] == i]
        data = age[age["Year"].isin(range(1990, 2015))]
        fert_list.append(data["Values"].mean())
    fert_data = fert_data[fert_data["Year"] == 1995]
    fert_data["Values"] = fert_list
    fert_data["Values"] = fert_data["Values"] / 2

    fert_func = si.splrep(fert_data["Age"], fert_data["Values"])

    #### AGE BIN CREATION
    # Calculate average fertility rate in each age bin using trapezoid
    # method with a large number of points in each bin.
    binsize = (max_yr - min_yr + 1) / totpers  # creating different generations (I believe?)

    num_sub_bins = float(10000)
    len_subbins = (np.float64(100 * num_sub_bins)) / totpers
    # 100 (lifetime year) / totpers gives us size of bins. To get length of subbin shouldnt we dividing by num_sub_bins ????
    age_sub = (np.linspace(np.float64(binsize) / num_sub_bins, # gives us the first subbin (len subbin)
                           np.float64(max_yr), # gives us end point
                           int(num_sub_bins*max_yr)) - 0.5 *  #
               np.float64(binsize) / num_sub_bins)
    # gives us mid age of all subbins

    ### POPULATION CREATION
    ages = np.linspace(min_yr, max_yr, curr_pop_pct.shape[0])
    pop_func = si.splrep(ages, curr_pop_pct)
    new_bins = np.linspace(min_yr, max_yr,\
                            num_sub_bins * max_yr)
    curr_pop_sub = si.splev(new_bins, pop_func)
    curr_pop_sub = curr_pop_sub / curr_pop_sub.sum()
    fert_rates_sub = np.zeros(curr_pop_sub.shape)
    pred_ind = (age_sub > fert_data["Age"].iloc[0]) * (age_sub < fert_data["Age"].iloc[-1])  # makes sure it is inside valid range
    age_pred = age_sub[pred_ind]  #gets age_sub in the valid range by applying pred_ind
    fert_rates_sub[pred_ind] = np.float64(si.splev(age_pred, fert_func))
    fert_rates_sub[fert_rates_sub < 0] = 0
    fert_rates = np.zeros(totpers)
    end_sub_bin = 0
    for i in range(totpers):
        beg_sub_bin = int(end_sub_bin)
        end_sub_bin = int(np.rint((i + 1) * len_subbins))
        fert_rates[i] = ((
            curr_pop_sub[beg_sub_bin:end_sub_bin] *
            fert_rates_sub[beg_sub_bin:end_sub_bin]).sum() /
            curr_pop_sub[beg_sub_bin:end_sub_bin].sum())
    fert_rates = np.nan_to_num(fert_rates)

    # this is basically finding the average fertility rate for each bin
    # (using interpolation to find the fertility rates for each subbins and find average of the bin )
    # we believe we should also get the interpolation of population!

    # this is for 1 year, we can include more years ! get most recent fertility data and assume constant over time

    if graph:
        '''
        ----------------------------------------------------------------
        age_fine_pred  = (300,) vector, equally spaced support of ages
                         between the minimum and maximum interpolating
                         ages
        fert_fine_pred = (300,) vector, interpolated fertility rates
                         based on age_fine_pred
        age_fine       = (300+some,) vector of ages including leading
                         and trailing zeros
        fert_fine      = (300+some,) vector of fertility rates including
                         leading and trailing zeros
        age_mid_new    = (totpers,) vector, midpoint age of each model
                         period age bin
        output_fldr    = string, folder in current path to save files
        output_dir     = string, total path of OUTPUT folder
        output_path    = string, path of file name of figure to be saved
        ----------------------------------------------------------------
        '''
        # Generate finer age vector and fertility rate vector for
        # graphing cubic spline interpolating function
        age_fine_pred = np.linspace(pop_data["Age"].iloc[0], pop_data["Age"].iloc[-1], 300)
        age_fine_pred[age_fine_pred > 50] = 50
        #fert_fine_pred = fert_func(age_fine_pred)
        fert_fine_pred = si.splev(age_fine_pred, fert_func)
        fert_fine_pred = np.nan_to_num(fert_fine_pred)
        fert_fine_pred[fert_fine_pred < 0] = 0
        age_fine = np.hstack((min_yr, age_fine_pred, max_yr))
        fert_fine = np.hstack((0, fert_fine_pred, 0))
        age_mid_new = (np.linspace(np.float(max_yr) / totpers, max_yr,
                                   totpers) - (0.5 * np.float(max_yr) /
                                               totpers))

        fig, ax = plt.subplots()
        plt.scatter(fert_data["Age"], fert_data["Values"], s=70, c='blue', marker='o',
                    label='Data')
        plt.scatter(age_mid_new, fert_rates, s=40, c='red', marker='d',
                    label='Model period (integrated)')
        plt.plot(age_fine, fert_fine, label='Cubic spline')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Fertility rate $f_{s}$')
        plt.xlim((min_yr - 1, max_yr + 1))
        plt.ylim((-0.15 * (fert_fine_pred.max()),
                  1.15 * (fert_fine_pred.max())))
        plt.legend(loc='upper right')
        #plt.text(-5, -0.018,
        #         "Source: National Vital Statistics Reports, " +
        #         "Volume 64, Number 1, January 15, 2015.", fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        # Create directory if OUTPUT directory does not already exist
        output_fldr = "OUTPUT/Demographics"
        output_dir = os.path.join(cur_path, output_fldr)
        if os.access(output_dir, os.F_OK) is False:
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "fert_rates")
        plt.savefig(output_path)

    return fert_rates

#%%
def select_data(data):
        new_data = data[data["Age"] != "110+"].astype(int)
        new_data = new_data[new_data["Year"] == 2014]
        return new_data
#%%
def get_mort(totpers, min_yr, max_yr, graph=False):
    # Get mortality rate by age data
    infmort_rate = 1.9 / 1000  # https://knoema.com/atlas/Japan/Infant-mortality-rate

    mort_filename = "/data/Mortality.csv"

    mort = pd.read_csv(mort_filename, sep="\s+", usecols=["Year", "Age", "Total"])
    mort = select_data(mort)[:110] # because don't know how to deal with 110

    pop_filename = "/data/Population.csv"
    pop = pd.read_csv(pop_filename, sep="\s+", usecols=["Year", "Age", "Total"])
    pop = select_data(pop)[:110]

    age_year_all = mort['Age'] + 1

    mort_rates_all = mort['Total'] / pop['Total']
    mort_rates_all = mort_rates_all[:110]

    # Calculate implied mortality rates in sub-bins of mort_rates_all.
    num_sub_bins = int(10000)
    len_subbins = ((np.float64((max_yr - min_yr + 1) * num_sub_bins)) /
                   totpers)  # length of a model period in data sub-bins

    # mortality rates by sub-bin implied by mort_rates_mxyr
    mort_func = si.splrep(mort["Age"] + 1, mort_rates_all)

    new_bins = np.linspace(min_yr, max_yr,\
                            len_subbins * totpers, dtype=float)

    mort_rates_sub_orig = 1 - si.splev(new_bins, mort_func)
    mort_rates_sub_orig[mort_rates_sub_orig > 1] = 1

    end_sub_bin = 0
    mort_rates_sub = np.zeros(mort_rates_sub_orig.shape)

    for i in range(min_yr, max_yr + 1):
        beg_sub_bin = int(end_sub_bin)
        end_sub_bin = int(np.rint((i + 1) * num_sub_bins))
        tot_period_surv = (np.log(mort_rates_sub_orig[beg_sub_bin:end_sub_bin]) ).sum()
        end_surv = np.log(1 - mort_rates_all.iloc[i])
        pow = end_surv / tot_period_surv
        mort_rates_sub[beg_sub_bin:end_sub_bin] = mort_rates_sub_orig[beg_sub_bin:end_sub_bin] ** pow

    mort_rates = np.zeros(totpers)
    end_sub_bin = 0
    for i in range(totpers):
        beg_sub_bin = int(end_sub_bin)
        end_sub_bin = int(np.rint((i + 1) * len_subbins))
        mort_rates[i] = 1 - mort_rates_sub[beg_sub_bin:end_sub_bin].prod()
    mort_rates[-1] = 1  # Mortality rate in last period is set to 1

    if graph:
        '''
        ----------------------------------------------------------------
        age_mid_new = (totpers,) vector, midpoint age of each model
                      period age bin
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of OUTPUT folder
        output_path = string, path of file name of figure to be saved
        ----------------------------------------------------------------
        '''
        age_mid_new = (np.linspace(np.float(max_yr) / totpers, max_yr,
                                   totpers) - (0.5 * np.float(max_yr) /
                                               totpers))
        fig, ax = plt.subplots()
        plt.scatter(np.hstack([0, age_year_all]),
                    np.hstack([infmort_rate, mort_rates_all]),
                    s=20, c='blue', marker='o', label='Data')
        plt.scatter(np.hstack([0, age_mid_new]),
                    np.hstack([infmort_rate, mort_rates]),
                    s=40, c='red', marker='d',
                    label='Model period (cumulative)')
        plt.plot(np.hstack([0, age_year_all[min_yr - 1:max_yr]]),
                 np.hstack([infmort_rate,
                            mort_rates_all[min_yr - 1:max_yr]]))
        plt.axvline(x=max_yr, color='red', linestyle='-', linewidth=1)
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Mortality rate $\rho_{s}$')
        plt.xlim((min_yr-2, age_year_all.max()+2))
        plt.ylim((-0.05, 1.05))
        plt.legend(loc='upper left')
        #plt.text(-5, -0.2,
        #         "Source: Actuarial Life table, 2011 Social Security " +
        #         "Administration.", fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        # Create directory if OUTPUT directory does not already exist
        output_fldr = "OUTPUT/Demographics"
        output_dir = os.path.join(cur_path, output_fldr)
        if os.access(output_dir, os.F_OK) is False:
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "mort_rates")
        plt.savefig(output_path)
        # plt.show()

    return mort_rates, infmort_rate

#%%
def pop_rebin(curr_pop_dist, totpers_new):
    totpers_orig = len(curr_pop_dist)
    if int(totpers_new) == totpers_orig:
        curr_pop_new = curr_pop_dist
    elif int(totpers_new) < totpers_orig:
        num_sub_bins = float(10000)
        ages = np.linspace(0, totpers_orig - 1, totpers_orig)
        pop_func = si.splrep(ages, curr_pop_dist)
        new_bins = np.linspace(0, totpers_orig - 1,\
                                num_sub_bins * totpers_orig)
        pop_ests = si.splev(new_bins, pop_func)
        len_subbins = ((np.float64(totpers_orig*num_sub_bins)) /
                       totpers_new)
        curr_pop_new = np.zeros(totpers_new, dtype=np.float64)
        end_sub_bin = 0
        for i in range(totpers_new):
            beg_sub_bin = int(end_sub_bin)
            end_sub_bin = int(np.rint((i + 1) * len_subbins))
            curr_pop_new[i] = \
                np.average(pop_ests[beg_sub_bin:end_sub_bin])
        # Return curr_pop_new to single precision float (float32)
        # datatype
        curr_pop_new = np.float32(curr_pop_new)

    return curr_pop_new
#%%
def select_immdata(data):
        new_data = data[data["Age"] != "110+"].astype(int)
        new_data = new_data[new_data["Year"] >= 2011]
        return new_data
#%%
def get_imm_resid(totpers, min_yr, max_yr, graph=True):
    pop_filename = "/data/Population.csv"

    pop_data = pd.read_csv(pop_filename, sep="\s+", usecols=["Year", "Age", "Total"])
    pop_data = select_immdata(pop_data)
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]

    pop_2010, pop_2011, pop_2012, pop_2013 = (
        np.array(pop_data_samp[pop_data_samp["Year"] == 2011]["Total"], dtype='f'),
        np.array(pop_data_samp[pop_data_samp["Year"] == 2012]["Total"], dtype='f'),
        np.array(pop_data_samp[pop_data_samp["Year"] == 2013]["Total"], dtype='f'),
        np.array(pop_data_samp[pop_data_samp["Year"] == 2014]["Total"], dtype='f'))

    pop_2010_EpS = pop_rebin(pop_2010, totpers)
    pop_2011_EpS = pop_rebin(pop_2011, totpers)
    pop_2012_EpS = pop_rebin(pop_2012, totpers)
    pop_2013_EpS = pop_rebin(pop_2013, totpers)
    # Create three years of estimated immigration rates for youngest age
    # individuals
    imm_mat = np.zeros((3, totpers))
    pop11vec = np.array([pop_2010_EpS[0], pop_2011_EpS[0],
                         pop_2012_EpS[0]])
    pop21vec = np.array([pop_2011_EpS[0], pop_2012_EpS[0],
                         pop_2013_EpS[0]])
    fert_rates = get_fert(totpers, min_yr, max_yr, False)
    mort_rates, infmort_rate = get_mort(totpers, min_yr, max_yr, False)
    newbornvec = np.dot(fert_rates, np.vstack((pop_2010_EpS,
                                               pop_2011_EpS,
                                               pop_2012_EpS)).T)
    imm_mat[:, 0] = ((pop21vec - (1 - infmort_rate) * newbornvec) /
                     pop11vec)
    # Estimate 3 years of immigration rates for all other-aged
    # individuals
    pop11mat = np.vstack((pop_2010_EpS[:-1], pop_2011_EpS[:-1],
                          pop_2012_EpS[:-1]))
    pop12mat = np.vstack((pop_2010_EpS[1:], pop_2011_EpS[1:],
                          pop_2012_EpS[1:]))
    pop22mat = np.vstack((pop_2011_EpS[1:], pop_2012_EpS[1:],
                          pop_2013_EpS[1:]))
    mort_mat = np.tile(mort_rates[:-1], (3, 1))
    imm_mat[:, 1:] = (pop22mat - (1 - mort_mat) * pop11mat) / pop12mat
    # Final estimated immigration rates are the averages over 3 years
    imm_rates = imm_mat.mean(axis=0)
    age_per = np.linspace(1, totpers, totpers)

    if graph:
        '''
        ----------------------------------------------------------------
        output_fldr = string, path of the OUTPUT folder from cur_path
        output_dir  = string, total path of OUTPUT folder
        output_path = string, path of file name of figure to be saved
        ----------------------------------------------------------------
        '''
        fig, ax = plt.subplots()
        plt.scatter(age_per, imm_rates, s=40, c='red', marker='d')
        plt.plot(age_per, imm_rates)
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'Age $s$ (model periods)')
        plt.ylabel(r'Imm. rate $i_{s}$')
        plt.xlim((0, totpers + 1))
        # Create directory if OUTPUT directory does not already exist
        output_fldr = "OUTPUT/Demographics"
        output_dir = os.path.join(cur_path, output_fldr)
        if os.access(output_dir, os.F_OK) is False:
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "imm_rates_orig")
        plt.savefig(output_path)
        # plt.show()

    return imm_rates
#%%
def immsolve(imm_rates, *args):
    fert_rates, mort_rates, infmort_rate, omega_cur_lev, g_n_SS = args
    omega_cur_pct = omega_cur_lev / omega_cur_lev.sum()
    totpers = len(fert_rates)
    OMEGA = np.zeros((totpers, totpers))
    OMEGA[0, :] = ((1 - infmort_rate) * fert_rates +
                   np.hstack((imm_rates[0], np.zeros(totpers-1))))
    OMEGA[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA[1:, 1:] += np.diag(imm_rates[1:])
    omega_new = np.dot(OMEGA, omega_cur_pct) / (1 + g_n_SS)
    omega_errs = omega_new - omega_cur_pct

    return omega_errs
#%%
def get_pop_objs(E, S, T, min_yr, max_yr, curr_year, GraphDiag=True):
    fert_rates = get_fert(E + S, min_yr, max_yr, graph=False)
    mort_rates, infmort_rate = get_mort(E + S, min_yr, max_yr,
                                        graph=False)
    mort_rates_S = mort_rates[-S:]
    imm_rates_orig = get_imm_resid(E + S, min_yr, max_yr, graph=False)
    OMEGA_orig = np.zeros((E + S, E + S))
    OMEGA_orig[0, :] = ((1 - infmort_rate) * fert_rates +
                        np.hstack((imm_rates_orig[0], np.zeros(E+S-1))))
    OMEGA_orig[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA_orig[1:, 1:] += np.diag(imm_rates_orig[1:])

    # Solve for steady-state population growth rate and steady-state
    # population distribution by age using eigenvalue and eigenvector
    # decomposition
    eigvalues, eigvectors = np.linalg.eig(OMEGA_orig)
    g_n_SS = (eigvalues[np.isreal(eigvalues)].real).max() - 1
    eigvec_raw =\
        eigvectors[:,
                   (eigvalues[np.isreal(eigvalues)].real).argmax()].real
    omega_SS_orig = eigvec_raw / eigvec_raw.sum()

    # Generate time path of the nonstationary population distribution
    omega_path_lev = np.zeros((E + S, T + S))
    pop_filename = "/data/Population.csv"

    pop_data = pd.read_csv(pop_filename, sep="\s+", usecols=["Year", "Age", "Total"])
    pop_data = select_immdata(pop_data)

    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    pop_2013 = np.array(pop_data_samp[pop_data_samp["Year"] == 2014]["Total"], dtype='f')
    # Generate the current population distribution given that E+S might
    # be less than max_yr-min_yr+1
    age_per_EpS = np.arange(1, E + S + 1)
    pop_2013_EpS = pop_rebin(pop_2013, E + S)
    pop_2013_pct = pop_2013_EpS / pop_2013_EpS.sum()
    # Age most recent population data to the current year of analysis
    pop_curr = pop_2013_EpS.copy()
    data_year = 2017 #Changed from 2013
    pop_next = np.dot(OMEGA_orig, pop_curr)
    g_n_curr = ((pop_next[-S:].sum() - pop_curr[-S:].sum()) /
                pop_curr[-S:].sum())  # g_n in 2013
    pop_past = pop_curr  # assume 2012-2013 pop
    # Age the data to the current year
    for per in range(curr_year - data_year):
        pop_next = np.dot(OMEGA_orig, pop_curr)
        g_n_curr = ((pop_next[-S:].sum() - pop_curr[-S:].sum()) /
                    pop_curr[-S:].sum())
        pop_past = pop_curr
        pop_curr = pop_next

    # Generate time path of the population distribution
    omega_path_lev[:, 0] = pop_curr.copy()
    for per in range(1, T + S):
        pop_next = np.dot(OMEGA_orig, pop_curr)
        omega_path_lev[:, per] = pop_next.copy()
        pop_curr = pop_next.copy()

    # Force the population distribution after 1.5*S periods to be the
    # steady-state distribution by adjusting immigration rates, holding
    # constant mortality, fertility, and SS growth rates
    imm_tol = 1e-14
    fixper = int(1.5 * S)
    omega_SSfx = (omega_path_lev[:, fixper] /
                  omega_path_lev[:, fixper].sum())
    imm_objs = (fert_rates, mort_rates, infmort_rate,
                omega_path_lev[:, fixper], g_n_SS)
    imm_fulloutput = opt.fsolve(immsolve, imm_rates_orig,
                                args=(imm_objs), full_output=True,
                                xtol=imm_tol)
    imm_rates_adj = imm_fulloutput[0]
    imm_diagdict = imm_fulloutput[1]
    omega_path_S = (omega_path_lev[-S:, :] /
                    np.tile(omega_path_lev[-S:, :].sum(axis=0), (S, 1)))
    omega_path_S[:, fixper:] = \
        np.tile(omega_path_S[:, fixper].reshape((S, 1)),
                (1, T + S - fixper))
    g_n_path = np.zeros(T + S)
    g_n_path[0] = g_n_curr.copy()
    g_n_path[1:] = ((omega_path_lev[-S:, 1:].sum(axis=0) -
                    omega_path_lev[-S:, :-1].sum(axis=0)) /
                    omega_path_lev[-S:, :-1].sum(axis=0))
    g_n_path[fixper + 1:] = g_n_SS
    omega_S_preTP = (pop_past.copy()[-S:]) / (pop_past.copy()[-S:].sum())
    imm_rates_mat = np.hstack((
        np.tile(np.reshape(imm_rates_orig[E:], (S, 1)), (1, fixper)),
        np.tile(np.reshape(imm_rates_adj[E:], (S, 1)), (1, T + S - fixper))))

    if GraphDiag:
        # Check whether original SS population distribution is close to
        # the period-T population distribution
        omegaSSmaxdif = np.absolute(omega_SS_orig -
                                    (omega_path_lev[:, T] /
                                     omega_path_lev[:, T].sum())).max()
        if omegaSSmaxdif > 0.0003:
            print("POP. WARNING: Max. abs. dist. between original SS " +
                  "pop. dist'n and period-T pop. dist'n is greater than" +
                  " 0.0003. It is " + str(omegaSSmaxdif) + ".")
        else:
            print("POP. SUCCESS: orig. SS pop. dist is very close to " +
                  "period-T pop. dist'n. The maximum absolute " +
                  "difference is " + str(omegaSSmaxdif) + ".")

        # Plot the adjusted steady-state population distribution versus
        # the original population distribution. The difference should be
        # small
        omegaSSvTmaxdiff = np.absolute(omega_SS_orig - omega_SSfx).max()
        if omegaSSvTmaxdiff > 0.0003:
            print("POP. WARNING: The maximimum absolute difference " +
                  "between any two corresponding points in the original"
                  + " and adjusted steady-state population " +
                  "distributions is" + str(omegaSSvTmaxdiff) + ", " +
                  "which is greater than 0.0003.")
        else:
            print("POP. SUCCESS: The maximum absolute difference " +
                  "between any two corresponding points in the original"
                  + " and adjusted steady-state population " +
                  "distributions is " + str(omegaSSvTmaxdiff))
        fig, ax = plt.subplots()
        plt.plot(age_per_EpS, omega_SS_orig, label="Original Dist'n")
        plt.plot(age_per_EpS, omega_SSfx, label="Fixed Dist'n")
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r"Pop. dist'n $\omega_{s}$")
        plt.xlim((0, E + S + 1))
        plt.legend(loc='upper right')
        # Create directory if OUTPUT directory does not already exist
        '''
        ----------------------------------------------------------------
        output_fldr = string, path of the OUTPUT folder from cur_path
        output_dir  = string, total path of OUTPUT folder
        output_path = string, path of file name of figure to be saved
        ----------------------------------------------------------------
        '''
        #cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "OUTPUT/Demographics"
        output_dir = os.path.join(cur_path, output_fldr)
        if os.access(output_dir, os.F_OK) is False:
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "OrigVsFixSSpop")
        plt.savefig(output_path)
        plt.show()

        # Print whether or not the adjusted immigration rates solved the
        # zero condition
        immtol_solved = \
            np.absolute(imm_diagdict['fvec'].max()) < imm_tol
        if immtol_solved:
            print("POP. SUCCESS: Adjusted immigration rates solved " +
                  "with maximum absolute error of " +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  ", which is less than the tolerance of " +
                  str(imm_tol))
        else:
            print("POP. WARNING: Adjusted immigration rates did not " +
                  "solve. Maximum absolute error of " +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  " is greater than the tolerance of " + str(imm_tol))

        # Test whether the steady-state growth rates implied by the
        # adjusted OMEGA matrix equals the steady-state growth rate of
        # the original OMEGA matrix
        OMEGA2 = np.zeros((E + S, E + S))
        OMEGA2[0, :] = ((1 - infmort_rate) * fert_rates +
                        np.hstack((imm_rates_adj[0], np.zeros(E+S-1))))
        OMEGA2[1:, :-1] += np.diag(1 - mort_rates[:-1])
        OMEGA2[1:, 1:] += np.diag(imm_rates_adj[1:])
        eigvalues2, eigvectors2 = np.linalg.eig(OMEGA2)
        g_n_SS_adj = (eigvalues[np.isreal(eigvalues2)].real).max() - 1
        if np.max(np.absolute(g_n_SS_adj - g_n_SS)) > 10 ** (-8):
            print("FAILURE: The steady-state population growth rate" +
                  " from adjusted OMEGA is different (diff is " +
                  str(g_n_SS_adj - g_n_SS) + ") than the steady-" +
                  "state population growth rate from the original" +
                  " OMEGA.")
        elif np.max(np.absolute(g_n_SS_adj - g_n_SS)) <= 10 ** (-8):
            print("SUCCESS: The steady-state population growth rate" +
                  " from adjusted OMEGA is close to (diff is " +
                  str(g_n_SS_adj - g_n_SS) + ") the steady-" +
                  "state population growth rate from the original" +
                  " OMEGA.")

        # Do another test of the adjusted immigration rates. Create the
        # new OMEGA matrix implied by the new immigration rates. Plug in
        # the adjusted steady-state population distribution. Hit is with
        # the new OMEGA transition matrix and it should return the new
        # steady-state population distribution
        omega_new = np.dot(OMEGA2, omega_SSfx)
        omega_errs = np.absolute(omega_new - omega_SSfx)
        print("The maximum absolute difference between the adjusted " +
              "steady-state population distribution and the " +
              "distribution generated by hitting the adjusted OMEGA " +
              "transition matrix is " + str(omega_errs.max()))

        # Plot the original immigration rates versus the adjusted
        # immigration rates
        immratesmaxdiff = \
            np.absolute(imm_rates_orig - imm_rates_adj).max()
        print("The maximum absolute distance between any two points " +
              "of the original immigration rates and adjusted " +
              "immigration rates is " + str(immratesmaxdiff))
        fig, ax = plt.subplots()
        print("Imm rates orig: ")
        print(imm_rates_orig)
        print("Imm rates adj:")
        print(imm_rates_adj)
        print("Equal:")
        print(imm_rates_orig == imm_rates_adj)
        plt.plot(age_per_EpS, imm_rates_orig, label="Original Imm. Rates", color="blue")
        plt.plot(age_per_EpS, imm_rates_adj, label="Adj. Imm. Rates", color="red")
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r"Imm. rates $i_{s}$")
        plt.xlim((0, E + S + 1))
        plt.legend(loc='upper center')
        # Create directory if OUTPUT directory does not already exist
        output_path = os.path.join(output_dir, "OrigVsAdjImm")
        plt.savefig(output_path)
        plt.show()

        # Plot population distributions for data_year, curr_year,
        # curr_year+20, omega_SSfx, and omega_SS_orig
        fig, ax = plt.subplots()
        plt.plot(age_per_EpS, pop_2013_pct, label="2013 pop.")
        plt.plot(age_per_EpS, (omega_path_lev[:, 0] /
                               omega_path_lev[:, 0].sum()),
                 label=str(curr_year) + " pop.")
        plt.plot(age_per_EpS, (omega_path_lev[:, int(0.5 * S)] /
                               omega_path_lev[:, int(0.5 * S)].sum()),
                 label="T=" + str(int(0.5 * S)) + " pop.")
        plt.plot(age_per_EpS, (omega_path_lev[:, int(S)] /
                               omega_path_lev[:, int(S)].sum()),
                 label="T=" + str(int(S)) + " pop.")
        plt.plot(age_per_EpS, omega_SSfx, label="Adj. SS pop.")
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r"Pop. dist'n $\omega_{s}$")
        plt.xlim((0, E+S+1))
        plt.legend(loc='upper right')
        # Create directory if OUTPUT directory does not already exist
        output_path = os.path.join(output_dir, "PopDistPath")
        plt.savefig(output_path)
        plt.show()

    # return omega_path_S, g_n_SS, omega_SSfx, survival rates,
    # mort_rates_S, and g_n_path
    return (omega_path_S.T, g_n_SS, omega_SSfx[-S:] /
            omega_SSfx[-S:].sum(), 1-mort_rates_S, mort_rates_S,
            g_n_path, imm_rates_mat.T, omega_S_preTP)
