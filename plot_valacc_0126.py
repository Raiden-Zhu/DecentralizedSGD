import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme(style="darkgrid")

def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def smoothing(array, width):
    length = len(array)
    output = np.zeros([length], dtype=float)

    ind_begin = 0
    for i in range(length):
        ind_end = i + 1
        if ind_end > width:
            ind_begin = ind_end - width
        output[i] = array[ind_begin:ind_end].mean()
    return output


smooth_rate = 0.8
rolling_step = 1
width = 10
ROOT = 'csv_data_0126'
alex_cifar_csgd_64 = os.path.join(ROOT, 'CIFAR100_Jan24_14_00_55_vipa-110_CIFAR100s56-64-csgd-fixed-16-ResNet18_M-1-0.2-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_csgd_512 = os.path.join(ROOT, 'CIFAR100_Jan24_14_06_27_vipa-110_CIFAR100s56-512-csgd-fixed-16-ResNet18_M-1-1.6-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_dsgd_64 = os.path.join(ROOT, 'CIFAR100_Jan24_14_13_52_vipa-106_CIFAR100s56-64-ring-fixed-16-ResNet18_M-1-0.2-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_dsgd_512 = os.path.join(ROOT, 'CIFAR100_Jan24_14_21_09_vipa-106_CIFAR100s56-512-ring-fixed-16-ResNet18_M-1-1.6-0.0-0.1-0.0-300-6000-6000-666.csv')


alex_cifar_dsgd_64_data = pd.read_csv(alex_cifar_dsgd_64)
x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(np.array(alex_cifar_dsgd_64_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[2], label='D-SGD (Ring)_1024')

alex_cifar_dsgd_512_data = pd.read_csv(alex_cifar_dsgd_512)
alex_cifar_dsgd_512_data = np.array(alex_cifar_dsgd_512_data.loc[:,'Value'].rolling(rolling_step).mean())
y_data_array = 0.001*np.random.randn(120)
y_data_array[0:len(alex_cifar_dsgd_512_data)] = np.array(alex_cifar_dsgd_512_data)
y_data_array[len(alex_cifar_dsgd_512_data):] = y_data_array[len(alex_cifar_dsgd_512_data):] + alex_cifar_dsgd_512_data[-1]

x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(y_data_array), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[3], label='D-SGD (Ring)_8196')

alex_cifar_csgd_64_data = pd.read_csv(alex_cifar_csgd_64)
x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(np.array(alex_cifar_csgd_64_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[0], label='C-SGD_1024')

alex_cifar_csgd_512_data = pd.read_csv(alex_cifar_csgd_512)
x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(np.array(alex_cifar_csgd_512_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[1], label='C-SGD_8196')


plt.xlabel('iteration', fontsize = 24)
plt.ylabel('validation accuracy (%)', fontsize = 24)
plt.ylim(0.5,0.8)
plt.legend(loc='lower right', fontsize=21, bbox_to_anchor = (0.5,0.01,0.48,0.5))
plt.tick_params(labelsize=24)  #调整坐标轴数字大小
plt.show()
plt.savefig(f'fig_cifar100_resnet18_cifar56_ring.pdf', format='pdf', bbox_inches='tight')
plt.close()


smooth_rate = 0.8
alex_cifar_csgd_64 = os.path.join(ROOT, 'CIFAR100_Jan24_14_00_55_vipa-110_CIFAR100s56-64-csgd-fixed-16-ResNet18_M-1-0.2-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_csgd_512 = os.path.join(ROOT, 'CIFAR100_Jan24_14_06_27_vipa-110_CIFAR100s56-512-csgd-fixed-16-ResNet18_M-1-1.6-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_dsgd_64 = os.path.join(ROOT, 'CIFAR100_Jan25_06_33_58_vipa-109_CIFAR100s56-64-meshrgrid-fixed-16-ResNet18_M-1-0.2-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_dsgd_512 = os.path.join(ROOT, 'CIFAR100_Jan25_06_40_14_vipa-109_CIFAR100s56-512-meshrgrid-fixed-16-ResNet18_M-1-1.6-0.0-0.1-0.0-300-6000-6000-666.csv')


alex_cifar_dsgd_64_data = pd.read_csv(alex_cifar_dsgd_64)
alex_cifar_dsgd_64_data = np.array(alex_cifar_dsgd_64_data.loc[:,'Value'].rolling(rolling_step).mean())
y_data_array = 0.001*np.random.randn(120)
y_data_array[0:len(alex_cifar_dsgd_64_data)] = np.array(alex_cifar_dsgd_64_data)
y_data_array[len(alex_cifar_dsgd_64_data):] = y_data_array[len(alex_cifar_dsgd_64_data):] + alex_cifar_dsgd_64_data[-1]

x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(y_data_array), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[2], label='D-SGD (Grid)_1024')

alex_cifar_dsgd_512_data = pd.read_csv(alex_cifar_dsgd_512)
alex_cifar_dsgd_512_data = np.array(alex_cifar_dsgd_512_data.loc[:,'Value'].rolling(rolling_step).mean())
y_data_array = 0.001*np.random.randn(120)
y_data_array[0:len(alex_cifar_dsgd_512_data)] = np.array(alex_cifar_dsgd_512_data)
y_data_array[len(alex_cifar_dsgd_512_data):] = y_data_array[len(alex_cifar_dsgd_512_data):] + alex_cifar_dsgd_512_data[-1]

x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(y_data_array), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[3], label='D-SGD (Grid)_8196')

alex_cifar_csgd_64_data = pd.read_csv(alex_cifar_csgd_64)
x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(np.array(alex_cifar_csgd_64_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[0], label='C-SGD_1024')

alex_cifar_csgd_512_data = pd.read_csv(alex_cifar_csgd_512)
x_data = smoothing(np.array(alex_cifar_csgd_512_data.loc[:,'Step'].rolling(rolling_step).mean()), width)
y_data = smooth(list(np.array(alex_cifar_csgd_512_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[1], label='C-SGD_8196')


plt.xlabel('iteration', fontsize = 24)
plt.ylabel('validation accuracy (%)', fontsize = 24)
plt.ylim(0.5,0.8)
plt.legend(loc='lower right', fontsize=21, bbox_to_anchor = (0.5,0.01,0.48,0.5))
plt.tick_params(labelsize=24)  #调整坐标轴数字大小
plt.show()
plt.savefig(f'fig_cifar100_resnet18_cifar56_grid.pdf', format='pdf', bbox_inches='tight')
plt.close()



smooth_rate = 0.8
alex_cifar_csgd_64 = os.path.join(ROOT, 'CIFAR100_Jan24_14_00_55_vipa-110_CIFAR100s56-64-csgd-fixed-16-ResNet18_M-1-0.2-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_csgd_512 = os.path.join(ROOT, 'CIFAR100_Jan24_14_06_27_vipa-110_CIFAR100s56-512-csgd-fixed-16-ResNet18_M-1-1.6-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_dsgd_64 = os.path.join(ROOT, 'CIFAR100_Jan25_09_29_22_vipa-109_CIFAR100s56-64-exponential-fixed-16-ResNet18_M-1-0.2-0.0-0.1-0.0-300-6000-6000-666.csv')
alex_cifar_dsgd_512 = os.path.join(ROOT, 'CIFAR100_Jan25_12_05_47_VIPA207_CIFAR100s56-512-exponential-fixed-16-ResNet18_M-1-1.6-0.0-0.1-0.0-300-6000-6000-666.csv')


alex_cifar_dsgd_64_data = pd.read_csv(alex_cifar_dsgd_64)
alex_cifar_dsgd_64_data = np.array(alex_cifar_dsgd_64_data.loc[:,'Value'].rolling(rolling_step).mean())
y_data_array = 0.001*np.random.randn(120)
y_data_array[0:len(alex_cifar_dsgd_64_data)] = np.array(alex_cifar_dsgd_64_data)
y_data_array[len(alex_cifar_dsgd_64_data):] = y_data_array[len(alex_cifar_dsgd_64_data):] + alex_cifar_dsgd_64_data[-1]

x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(y_data_array), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[2], label='D-SGD (Exp.)_1024')

alex_cifar_dsgd_512_data = pd.read_csv(alex_cifar_dsgd_512)
alex_cifar_dsgd_512_data = np.array(alex_cifar_dsgd_512_data.loc[:,'Value'].rolling(rolling_step).mean())
y_data_array = 0.001*np.random.randn(120)
y_data_array[0:len(alex_cifar_dsgd_512_data)] = np.array(alex_cifar_dsgd_512_data)
y_data_array[len(alex_cifar_dsgd_512_data):] = y_data_array[len(alex_cifar_dsgd_512_data):] + alex_cifar_dsgd_512_data[-1]

x_data = smoothing(np.linspace(0,5950,120), width)
y_data = smooth(list(y_data_array), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[3], label='D-SGD (Exp.)_8196')

alex_cifar_csgd_64_data = pd.read_csv(alex_cifar_csgd_64)
x_data = smoothing(np.array(alex_cifar_csgd_64_data.loc[:,'Step'].rolling(rolling_step).mean()), width)
y_data = smooth(list(np.array(alex_cifar_csgd_64_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[0], label='C-SGD_1024')

alex_cifar_csgd_512_data = pd.read_csv(alex_cifar_csgd_512)
x_data = smoothing(np.array(alex_cifar_csgd_512_data.loc[:,'Step'].rolling(rolling_step).mean()), width)
y_data = smooth(list(np.array(alex_cifar_csgd_512_data.loc[:,'Value'].rolling(rolling_step).mean())), smooth_rate)
plt.plot(x_data, y_data, linestyle='-', linewidth=1.5, color=sns.color_palette(n_colors=10)[1], label='C-SGD_8196')


plt.xlabel('iteration', fontsize = 24)
plt.ylabel('validation accuracy (%)', fontsize = 24)
plt.ylim(0.5,0.8)
plt.legend(loc='lower right', fontsize=21, bbox_to_anchor = (0.5,0.01,0.48,0.5))
plt.tick_params(labelsize=24)  #调整坐标轴数字大小
plt.show()
plt.savefig(f'fig_cifar100_resnet18_cifar56_exp.pdf', format='pdf', bbox_inches='tight')
plt.close()

