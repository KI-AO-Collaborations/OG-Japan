#%%
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.interpolate as si

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

# print('Values:')
# get_values(a0,a1,a2,a3,a4)

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
plt.legend(loc='lower left')
plt.scatter(labels[:-2], model_moments[:-2], color = 'b', label = r'Model Moments')
plt.legend(loc='lower left')
plt.scatter(labels[-2:], model_moments[-2:], color = 'g', label = r'Uncalibrated Model Moments')
plt.legend(loc='lower left')
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("labor_moments.png")
plt.show()
plt.close()

#%%
# new chi_n calibration:
labor_hours = np.array([167, 165, 165, 165, 165, 166, 165, 165, 164])#, 166, 164])
labor_part_rate = np.array([0.69, 0.849, 0.849, 0.847, 0.847, 0.859, 0.859, 0.709, 0.709])#, 0.212, 0.212])
employ_rate = np.array([0.937, 0.954, 0.954, 0.966, 0.966, 0.97, 0.97, 0.968, 0.968])#, 0.978, 0.978])
labor_hours_adj = labor_hours * labor_part_rate * employ_rate
labor_moments = labor_hours_adj * 12 / (365 * 17.5)
data_moments_trunc = np.array(list(labor_moments.flatten()))
model_moments = [0.22590727491968024, 0.25431586945815365, 
0.23899521787783148, 0.23480598940322048, 0.23129310947391893, 
0.23390337806027334, 0.2307653823166511, 0.18977931340477963, 
0.1919241624948257, 0.12291577097771472, 0.05630052613129159] # used now !
ages = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60]) + 2.5
ages_new = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 83]) + 2.5
plt.scatter(ages, labor_moments, color = 'b', label = r'Data Labor Supply')
plt.legend(loc='lower center')
labor_fun = si.splrep(ages, data_moments_trunc)
ages_full = np.linspace(21, 65, 45)
plt.scatter(ages_new, model_moments)
data_moments = si.splev(ages_full, labor_fun)
data_moments = np.repeat(data_moments_trunc, 5)
plt.xlabel(r'Age $s$')
plt.ylabel(r'Labor Supply $\frac{\bar{n_s}}{\tilde{l}}$')
plt.plot(ages_full, data_moments, color = 'r', label = r'Interpolated Labor Supply')
plt.legend(loc='lower center')
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("labor_interp_moments.png")
plt.show()
plt.close()

#%%


#%%
labor_model_moments = [0.1505088,  0.17247978, 0.19173943, 0.20384064, 0.21440464, 0.22152762,
 0.22673891, 0.22839788, 0.22918112, 0.22999457, 0.22787117, 0.22733152,
 0.22684068, 0.22703397, 0.22750995, 0.22849704, 0.22876223, 0.22900503,
 0.22921177, 0.22856699, 0.22818275, 0.22791578, 0.22799495, 0.22832315,
 0.2299042,  0.23016361, 0.23144241, 0.23280788, 0.23469094, 0.23517582,
 0.2348437,  0.23260581, 0.22806237, 0.22129255, 0.21391063, 0.20970673,
 0.20365511, 0.19579935, 0.18634477, 0.18079557, 0.18481301, 0.18815781,
 0.24743148, 0.26094669, 0.2383649]
labor_model_moments = [0.17280152, 0.20534884, 0.21147639, 0.235289,   0.15444407, 0.24695376,
 0.24446594, 0.36640391, 0.24128191, 0.2417963,  0.36377637, 0.36090858,
 0.2265956,  0.22555885, 0.22388283, 0.37432606, 0.37387789, 0.2589133,
 0.25978143, 0.26084655, 0.26193238, 0.26317574, 0.25433178, 0.25469551,
 0.25553841, 0.26156649, 0.26460427, 0.26171686, 0.26143935, 0.26128624,
 0.25956558, 0.27575473, 0.27309331, 0.260688,   0.25954435, 0.1522682,
 0.22007837, 0.340535,   0.22563522, 0.21350816, 0.22264555, 0.222299,
 0.28487272, 0.18735744, 0.27701419]
labor_model_moments = [0.25633439, 0.20093815, 0.21150954, 0.14738601, 0.2470485,  0.25100865,
 0.2511992,  0.23955692, 0.25376025, 0.25342003, 0.23735469, 0.23313953,
 0.36218893, 0.35912872, 0.35786861, 0.24355104, 0.24288904, 0.26385762,
 0.26479566, 0.26605243, 0.26743212, 0.1686283,  0.25984173, 0.25994932,
 0.26027909, 0.26672784, 0.16758965, 0.26689141, 0.26675111, 0.26672056,
 0.26514682, 0.26366972, 0.26478251, 0.26533119, 0.26478102, 0.24763209,
 0.22415406, 0.21668019, 0.2165665,  0.31061041, 0.21997966, 0.219987,
 0.17995947, 0.30301344, 0.17490099]
labor_model_moments = [0.20878565, 0.1975463,  0.2024006,  0.19690515, 0.20710516, 0.25753327,
 0.24368167, 0.24587537, 0.24728257, 0.24453552, 0.24353383, 0.24319411,
 0.24326923, 0.24390919, 0.24492123, 0.24645355, 0.24714776, 0.24788457,
 0.24855516, 0.24835614, 0.24847061, 0.24876036, 0.24939762, 0.25036787,
 0.25267278, 0.25341386, 0.25523224, 0.25731923, 0.25928462, 0.26046565,
 0.2600806,  0.25763767, 0.25242267, 0.2484289,  0.25953835, 0.21015545,
 0.20365892, 0.21476219, 0.20367817, 0.21676054, 0.22109761, 0.20413805,
 0.20310546, 0.21396072, 0.21728373]

plt.xlabel(r'Age $s$')
plt.ylabel(r'Labor Supply $\frac{\bar{n_s}}{\tilde{l}}$')
ages = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60]) + 2.5
labor_fun = si.splrep(ages, data_moments_trunc)
ages_full = np.linspace(21, 65, 45)
data_moments = np.repeat(data_moments_trunc, 5)
plt.plot(ages_full, data_moments, color = 'r', label = r'Interpolated Data Moments')
plt.legend(loc='lower center')
plt.scatter(ages_full, labor_model_moments, color = 'b', label = r'Model Moments')
plt.legend(loc='lower center')
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("labor_trial1.png")
plt.show()
plt.close()

#%%
# Plot Chi_n
chi_n_values = [109.86719764, 106.06662648, 97.53350752, 98.67194012, 89.18637004,
  63.63262154,  68.73372453, 68.08110162, 67.99462187, 69.9290688,
  71.28638356, 72.21669868, 73.14630267, 74.05352534, 74.77536528,
  75.29121172, 76.31856054, 77.30599801, 78.4446991, 79.82121387,
  81.07389146, 82.11311563, 82.88768804, 83.44404256, 83.26481037,
  83.69296118, 83.50898674, 83.13352735, 82.38135449, 82.33264806,
  82.69450477, 83.91409057, 86.3669977, 88.13816005, 82.19960535,
 111.36520435, 115.46772834, 105.47260457, 112.29498048, 100.777425,
  95.88893856, 105.2745885, 103.238911, 92.76672994, 87.57323408,
 133.61104417, 147.37526863, 161.1394931, 174.90371757, 188.66794204,
 202.4321665, 216.19639097, 229.96061544, 243.7248399, 257.48906437,
 271.25328884, 285.0175133, 298.78173777, 312.54596224, 326.31018671,
 340.07441117, 353.83863564, 367.60286011, 381.36708457, 395.13130904,
 408.89553351, 422.65975798, 436.42398244, 450.18820691, 463.95243138,
 477.71665584, 491.48088031, 505.24510478, 519.00932925, 532.77355371,
 546.53777818, 560.30200265, 574.06622711, 587.83045158, 601.59467605]

ages_full = np.linspace(21, 100, 80)
print(len(ages_full))
print(len(chi_n_values))
plt.scatter(ages_full, chi_n_values)
plt.show()
plt.close()

ages_full = np.linspace(21, 65, 45)
plt.scatter(ages_full, chi_n_values[:45])
plt.show()
plt.close()

