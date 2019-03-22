#%%
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
import os

#%%
def MU_sumsq(ellip_params, *args):
   
    b_ellip, upsilon = ellip_params
    Frisch, l_tilde, labor_sup = args

    MU_CFE = (labor_sup ** (1 / Frisch))
    MU_ellip = ((b_ellip / l_tilde) *
                ((labor_sup / l_tilde) ** (upsilon - 1)) *
                ((1 - ((labor_sup / l_tilde) ** upsilon)) **
                ((1 - upsilon) / upsilon)))
    sumsq = ((MU_ellip - MU_CFE) ** 2).sum()

    return sumsq    


def fit_ellip_CFE(ellip_init, cfe_params, l_tilde, graph):

    Frisch = cfe_params
    labor_min = 0.05
    labor_max = 0.95 * l_tilde
    labor_N = 1000
    labor_sup = np.linspace(labor_min, labor_max, labor_N)
    fit_args = (Frisch, l_tilde, labor_sup)
    bnds_elp = ((1e-12, None), (1 + 1e-12, None))
    ellip_params = opt.minimize(
        MU_sumsq, ellip_init, args=(fit_args), method='L-BFGS-B',
        bounds=bnds_elp)
    print(ellip_params)
    b_ellip, upsilon = ellip_params.x
    sumsq = ellip_params.fun
    if ellip_params.success:
        print('SUCCESSFULLY ESTIMATED ELLIPTICAL UTILITY.')
        print('b=', b_ellip, ' upsilon=', upsilon, ' SumSq=', sumsq)

        if graph:

            # Plot steady-state consumption and savings distributions
            MU_ellip = \
                ((b_ellip / l_tilde) *
                 ((labor_sup / l_tilde) ** (upsilon - 1)) *
                 ((1 - ((labor_sup / l_tilde) ** upsilon)) **
                 ((1 - upsilon) / upsilon)))
            MU_CFE = (labor_sup ** (1 / Frisch))
            fig, ax = plt.subplots()
            plt.plot(labor_sup, MU_ellip, label='Elliptical MU')
            plt.plot(labor_sup, MU_CFE, label='CFE MU')
            # for the minor ticks, use no labels; default NullFormatter
            minorLocator = MultipleLocator(1)
            ax.xaxis.set_minor_locator(minorLocator)
            plt.grid(b=True, which='major', color='0.65', linestyle='-')
            plt.xlabel(r'Labor supply $n_{s,t}$')
            plt.ylabel(r'Marginal disutility')
            plt.xlim((0, l_tilde))
            # plt.ylim((-1.0, 1.15 * (b_ss.max())))
            plt.legend(loc='upper left')
            plt.tight_layout(rect=(0, 0.03, 1, 1))
            plt.savefig("cfe_calibrate.png")
            plt.close()
    else:
        print('NOT SUCCESSFUL ESTIMATION OF ELLIPTICAL UTILITY')

    return b_ellip, upsilon


#%%
b_ellip = 0.5
upsilon = 1.5
ellip_init = np.array([b_ellip, upsilon])
cfe_params = 0.5
l_tilde = 1 #17.5 / 24
fit_ellip_CFE(ellip_init, cfe_params, l_tilde, True)



