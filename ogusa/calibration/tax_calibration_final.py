#%%
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#%%
incomes = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700 , 1800, 1900, 2000]])
incomes = incomes * 10000
# URL: https://www.mf-realty.jp/tebiki/taxbusiness/capter06_05.html
effective_tax = np.array([0.156, 0.164, 0.172, 0.21, 0.238, 0.258, 0.272, 0.286, 0.297, 0.316, 0.331, 0.344, 0.355, 0.364, 0.373, 0.38, 0.386, 0.392, 0.40, 0.48])

#%%
### GS Tax Function
# URL: https://www.jstor.org/stable/pdf/41789070.pdf
def tax_func(I, phi0, phi1, phi2):
    #txrates = ((phi0 * (I - ((I ** -phi1) + phi2) ** (-1 / phi1))) / I)
    txrates = phi0 - phi0 * (phi1 * I ** phi2 + 1)**(-1 / phi2)
    return txrates

def tax_func2(I, phi0, phi1, phi2):
    tau = (phi0 * (I - ((I ** -phi1) + phi2) ** (-1 / phi1))) 
    return tau 

#%%
def model_moments(I_array, phi0, phi1, phi2):
    return tax_func(I_array, phi0, phi1, phi2)

def data_moments():
    effective_tax = np.array([0.156, 0.164, 0.172, 0.21, 0.238, 0.258, 0.272, 0.286, 0.297, 0.316, 0.331, 0.344, 0.355, 0.364, 0.373, 0.38, 0.386, 0.392, 0.40, 0.48])
    return effective_tax

def err_vec(income, phi0, phi1, phi2, simple):
    
    data_mms = data_moments()
    model_mms = model_moments(income, phi0, phi1, phi2)
    
    if simple:
        err_vec = model_mms - data_mms
    else:
        err_vec = (model_mms - data_mms) / data_mms
    
    return err_vec

def criterion(params, *args):
    
    phi0, phi1, phi2 = params
    income, W = args
    
    err = err_vec(income, phi0, phi1, phi2, simple = False).squeeze()
    crit_val = err.T @ W @ err
    return crit_val

#%%
### Optimization Problem: 
# Initial guess of parameters
phi0 = 0.479
phi1 = 0.022
phi2 = 0.817
params_init = np.array([phi0, phi1, phi2])

# Weighting matrix 
W_hat = np.eye(20)

incomes = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700 , 1800, 1900, 2000]])
incomes = incomes * 10000
incomes = incomes * 10 ** (-6)
#gmm_args = (I_array, I_array_2, W_hat)
gmm_args = (incomes, W_hat)

# Optimization
results_GMM = opt.minimize(criterion, params_init, args = (gmm_args), method = 'L-BFGS-B')
print(results_GMM)
phi0_GMM, phi1_GMM, phi2_GMM = results_GMM.x

#%%
### Plots
I = np.linspace(1,20,20)
tax_rate = tax_func(I, phi0_GMM, phi1_GMM, phi2_GMM)
plt.xlabel('Income (Millions of Yen)')
plt.ylim(0, 0.5)
plt.ylabel('Effecitve Tax Rate $\tau_{s,t}^{ETR}$')
plt.plot(I, tax_rate, color = 'r', label = r'Estimated Tax Rates')
plt.legend(loc='upper left')

tax_rate_data = np.array(effective_tax)
plt.scatter(I, tax_rate_data, label = r'Calculated Tax Rates')
plt.legend(loc='upper left')

plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("effective_tax_gs.png")
plt.close()
#%%
def marg_tax(I, phi0, phi1, phi2):
    margrates = phi0 * phi1 * I ** (phi2 - 1) * (phi1 * I ** phi2 + 1) ** ( (- 1 - phi2) / phi2)
    return margrates

marg = marg_tax(I, phi0_GMM, phi1_GMM, phi2_GMM)
I = np.linspace(1,20,20)
plt.xlabel('Income (Millions of Yen)')
plt.ylabel(r'Marginal Tax Rate $\tau_{s,t}^{MTR}$')
plt.plot(I, marg, color = 'r', label = r'Estimated Tax Rates')
plt.legend(loc='upper right')
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("marginal_tax_gs.png")
plt.close()

