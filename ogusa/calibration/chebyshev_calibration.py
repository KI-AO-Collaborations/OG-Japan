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
    ages = np.linspace(20, 65, 46)
    return np.polynomial.chebyshev.chebfit(ages, data_moments, 4)

print('Values:')
get_values(a0,a1,a2,a3,a4)

chi_values = np.array([97.18375544, 84.29385451, 73.28369617, 63.97171784, 56.18503188, #25
       49.75942557, 44.53936113, 40.37797572, 37.13708141, 34.68716522, #30
       32.90738912, 31.68558998, 30.91827963, 30.51064481, 30.37654721, #35
       30.43852346, 30.6277851 , 30.88421861, 31.15638543, 31.40152189, #40
       31.58553929, 31.68302385, 31.67723671, 31.56011397, 31.33226664, #45
       31.00298067, 30.59021696, 30.12061131, 29.62947449, 29.16079218, #50
       28.76722499, 28.51010849, 28.45945315, 28.69394439, 29.30094257, #55
       30.37648298, 32.02527583, 34.36070627, 37.5048344 , 41.58839522, #60
       46.7507987 , 53.14012972, 60.9131481 , 70.23528858, 81.28066087,
       94.23204957])
chi_values[:5] += 0# Ages 20-24
chi_values[5:10] += 0 # Ages 25-29
chi_values[10:15] += 3# Ages 30-34
chi_values[15:20] += 4 # Ages 35-39
chi_values[20:25] += 5 # Ages 40-44
chi_values[25:30] += 7 # Ages 45-49
chi_values[30:35] += 8 # Ages 50-54
chi_values[35:40] += 10# Ages 55-59
chi_values[40:45] -= 3 # Ages 60-64

chi_values += 20
chi_values
print("Params:")
get_chebyshev_params(chi_values)



#%%
# Chi_n Graph
a0 = 1.10262456e+03
a1 = -1.02616544e+02
a2 = 1.84708517e+00
a3 = -1.44960570e-02
a4 = 4.19655083e-05

a0_new = 1.10272456e+03
a1_new = -1.02616544e+02
a2_new = 1.84708517e+00
a3_new = -1.44960570e-02
a4_new = 4.19655083e-05

# Plot of Chi_n
ages_beg = np.linspace(20, 65, 46)
data_beg = func(ages_beg, a0, a1,a2,a3,a4)
data_beg_new = func(ages_beg, a0_new, a1_new, a2_new, a3_new, a4_new)
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
plt.show()
plt.close()

plt.scatter(ages_beg, chi_values) #data_points
plt.plot(ages_beg, data_beg)
plt.show()
plt.close()

# Labor model_moments from Simulation:
model_moments = np.array([0.2040540765626132, 0.2526354466647778, 0.2769057405403771, 0.27696186745847984, 0.27235828933517375, 0.27723433376871154, 0.2881355407467818, 0.2541506887816195, 0.1789017426314303, 0.11635354318881265, 0.059785018218161144])
model_moments = np.array([0.1499162057778458, 0.22552815020838995, 0.26996153865471983, 0.2691089631249235, 0.25935860093557483, 0.2560840007407618, 0.2423648503430985, 0.18459865014459426, 0.1062297896818786, 0.0058177805287718135, 0.0011070846641351186])
model_moments = np.array([0.19764733496982848, 0.2436643741628969, 0.27197514385533444, 0.2803143580895792, 0.28810678688089614, 0.3010253299597592, 0.30601085556631824, 0.27005934137685045, 0.19058363065782624, 0.12137536035707344, 0.05488615185261614])
model_moments = np.array([0.1979201105331154, 0.24450670048502157, 0.2667324002094316, 0.26787924076936975, 0.27142440961625264, 0.2834758854880253, 0.2924960584176072, 0.2641984077417718, 0.18845689178864306, 0.11965783130836437, 0.053956744146367544])
model_moments = np.array([0.1896358996671081, 0.23201411990561313, 0.2527955933722157, 0.2514736466065953, 0.24942907166398584, 0.25910371394453613, 0.26245265364074316, 0.2398173278384821, 0.18185968256973795, 0.12215688804408117, 0.05829827316636392])
labels = np.linspace(20, 70, 11)
labels += 2
labels[-1] = 85


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
plt.show()
plt.close()
