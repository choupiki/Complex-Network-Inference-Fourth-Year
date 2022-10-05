import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import data_namer 
import csv
import argparse
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from time import perf_counter
from tqdm import tqdm
import pandas as pd
import winhelpers_SF as whelp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics
import scipy as sp

parser = argparse.ArgumentParser()
parser.add_argument('--indir', nargs='?', required=True)
#parser.add_argument('--outdir', nargs='?', required=True)
#parser.add_argument('--nJobs', type=int, required=True)
args = parser.parse_args()
indir = args.indir


df = pd.read_csv(str(indir) + '.csv', delimiter=':')
#df_r = pd.read_csv(str(indir) + '_ranges.csv', delimiter=':')

df = df.sort_values(by=['T', 'gamma'])
df
T = np.array(df.T)
T = T[:][4]
gamma = np.array(df.gamma)
n = np.array(df.n)
r_err = df.r_err
mag = df.mag
sus = df.sus
#no_h_err = df.no_h_err


print(df, 'n\n\n T:', T, 'n\n\n', gamma, 'n\n\n', mag, 'n\n\n', r_err, 'n\n\n', sus, 'n\n\n')

print(T[-1])

T_range = whelp.param_range([T[0], T[-1]], n[0])
g_range = whelp.param_range([gamma[0], gamma[-1]], n[0])
print(T_range)
print(g_range)
print(len(T_range))
"""
with open('tempranges.csv', 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    # writing the fields 
    #csvwriter.writerow(('T_range', 'gamma_range'))  
    # writing the data rows 
    csvwriter.writerow(T_range)
"""
r_err_med = statistics.median(r_err)
#no_h_err_med = statistics.median(no_h_err)
print('r_err median: \n\n', r_err_med)
#print('r_err median: \n\n', no_h_err_med)


plot_mag = np.zeros((len(T_range), len(g_range)))
plot_r_err = np.zeros((len(T_range), len(g_range)))
plot_sus = np.zeros((len(T_range), len(g_range)))
#plot_no_h_err = np.zeros((len(T_range), len(g_range)))
for i in range(len(T_range)):
    for ii in range(len(g_range)):
        j = i*len(g_range) + ii
        plot_mag[i, ii] = mag.iloc[j]
        plot_sus[i, ii] = sus.iloc[j]
        if r_err.iloc[j] > 3*r_err_med:
            plot_r_err[i, ii] = r_err_med
            #plot_no_h_err[i, ii] = no_h_err_med
        else:     
            plot_r_err[i, ii] = r_err.iloc[j]
            #plot_no_h_err[i, ii] = no_h_err.iloc[j]

# Plotting
extent = [T[0] , T[-1], gamma[-1], gamma[0]]

fig, axs = plt.subplots(2, 2, sharey=True)

#fig.suptitle('Title of figure', fontsize=20)
T_c = []
for i in range(len(g_range)):
    if g_range[i] > 3.005:
        T_c.append(-2/(np.log((4-g_range[i])/(g_range[i]-2))))
    else:
        T_c.append(7)
        
axs[0, 0].set_title('Absolute Value of Magnetisation')
axs[0, 0].set_xlabel('T')
axs[0, 0].plot(T_c, g_range, linewidth=2, color='r')
axs[0, 0].set_ylabel('Gamma', fontweight='bold')
im1 = axs[0, 0].imshow(sp.ndimage.gaussian_filter(plot_mag, 2), extent=extent, aspect='auto')
divider1 = make_axes_locatable(axs[0, 0])
cax1 = divider1.append_axes("right", size="20%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1, label='Magnetisation Strength')

axs[1, 0].set_title('Reconstruction Error')
axs[1, 0].set_xlabel('T')
axs[1, 0].plot(T_c, g_range, linewidth=2, color='r')
im2 = axs[1, 0].imshow(sp.ndimage.gaussian_filter(plot_r_err, 2), extent=extent, aspect='auto')
divider2 = make_axes_locatable(axs[1, 0])
cax2 = divider2.append_axes("right", size="20%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2, label='Magnitutde of Error')

axs[0, 1].set_title('Susceptibility')
axs[0, 1].set_xlabel('T')
im3 = axs[0, 1].imshow(sp.ndimage.gaussian_filter(plot_sus, 2), extent=extent, aspect='auto')
divider3 = make_axes_locatable(axs[0, 1])
cax3 = divider3.append_axes("right", size="20%", pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3, label='Magnitutde of Chi')

plt.tight_layout()
# Make space for title
#plt.subplots_adjust(top=0.85)
plt.show()





