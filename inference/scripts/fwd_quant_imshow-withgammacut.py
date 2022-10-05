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
import math

# Parse args to program
parser = argparse.ArgumentParser()
parser.add_argument('--indir', nargs='?', required=True)
parser.add_argument('--tolerance', type=float, nargs='?', required=True)
parser.add_argument('--blur_sig', type=float, nargs='?', required=True)
args = parser.parse_args()
indir = args.indir
# Create Dataframe
df = pd.read_csv(str(indir) + '.csv', delimiter=':')

    
# Complete/2D Ising Plots
    
# Scale-Free Plots

df = df.sort_values(by=['T', 'gamma'])
df
# Read results and facility params from dataframe
T = np.array(df.T)
T = T[:][4]
n = np.array(df.n)
r_err = df.r_err
N = np.array(df.N)
#k_0 = np.array(df.k_0)
mag = df.mag
sus = df.sus
no_h_err = df.no_h_err
gamma = np.array(df.gamma)
print(df, 'n\n\n T:', T, 'n\n\n', gamma, 'n\n\n', mag, 'n\n\n', r_err, 'n\n\n', sus, 'n\n\n')
print(T[-1])
T_range = whelp.param_range([T[0], T[-1]], n[0])
"""
if math.isnan(gamma[-1]):
    gamma[-1] = gamma[-2] + (gamma[1] - gamma[0])
else:
    pass
"""
print(gamma[-1])
g_range = whelp.param_range([gamma[0], gamma[-1]], n[0])
print(T_range)
print(g_range)
r_err_med = statistics.median(r_err)
no_h_err_med = statistics.median(no_h_err)
sus_med = np.max(sus) - args.tolerance*10*statistics.median(sus)
print('r_err median: \n\n', r_err_med)
print('no_h_err median: \n\n', no_h_err_med)
print('sus median: \n\n', sus_med)

plot_mag = np.zeros((len(T_range), len(g_range)))
plot_r_err = np.zeros((len(T_range), len(g_range)))
plot_sus = np.zeros((len(T_range), len(g_range)))
plot_no_h_err = np.zeros((len(T_range), len(g_range)))
for i in range(len(T_range)):
    for ii in range(len(g_range)):
        j = i*len(g_range) + ii
        plot_mag[i, ii] = mag.iloc[j]
        if sus.loc[j] > args.tolerance*sus_med:
            plot_sus[i, ii] = args.tolerance*sus_med
        else:
            plot_sus[i, ii] = sus.iloc[j]
        if r_err.iloc[j] > args.tolerance*r_err_med:
            plot_r_err[i, ii] = args.tolerance*r_err_med
            plot_no_h_err[i, ii] = args.tolerance*no_h_err_med
        else:     
            plot_r_err[i, ii] = r_err.iloc[j]
            plot_no_h_err[i, ii] = no_h_err.iloc[j]
max_err = np.max(plot_r_err)
min_err = np.min(plot_r_err)
var_err = np.var(plot_r_err)          
print(max_err, '\n', min_err)

#max_sus = np.zeros((len(T_range), len(g_range)))
gam_min_err = []
gam_max_sus = []
i = 0
for i in range(len(g_range)):
    ##max_sus[i, np.argmax(plot_sus[i, :])] = plot_sus[i, np.argmax(plot_sus[i, :])]
    gam_max_sus.append(T_range[np.argmax(plot_sus[i, :])])
    gam_min_err.append(T_range[np.argmin(plot_r_err[i, :])])



# Plotting
extent = [T[0], T[-1], gamma[-1], gamma[0]]
fig, axs = plt.subplots(1, 2, sharey=True)
"""
if hasattr(df, 'model') == False:
    fig.suptitle('Scale Free Results', fontsize=14)
if hasattr(df, 'model') == True:
    fig.suptitle(df.model[0], fontsize=14)
"""
T_c = []
T_c_2 = []
for i in range(len(g_range)):
    if g_range[i] > 3.005:
        T_c.append(-2/(np.log((4-g_range[i])/(g_range[i]-2))))
    else:
        T_c.append(7)
      
for i in range(len(g_range)):
    if g_range[i] < 2.25:
        T_c_2.append(0)
    if g_range[i] < 3 and g_range[i] >= 2.25:
        T_c_2.append(((g_range[i]-2)*N[0]**((3-g_range[i])*(g_range[i]-1))*np.mean(1))/(3-g_range[i]))
    if g_range[i] == 3:
        T_c_2.append((np.log(N)*(g_range[i]-1)*np.mean(1))/(4*(g_range[i]-2)))
    if g_range[i] > 3 and g_range[i] <= 4:
        T_c_2.append(-2/(np.log((4-g_range[i])/(g_range[i]-2))))
    if g_range[i] > 4:
        T_c_2.append(0)

print(T_c_2)       
axs[0].set_title('Absolute Value of Magnetisation')
axs[0].set(ylim=[4, 2.5])
axs[0].set_xlabel('T')
axs[0].plot(T_c_2, g_range, linewidth=2, color='r')
axs[0].set_ylabel('Gamma', fontweight='bold')
axs[0].plot(sp.ndimage.gaussian_filter1d(gam_max_sus, args.blur_sig), g_range, linewidth=1, color='cyan')
axs[0].plot(sp.ndimage.gaussian_filter1d(gam_min_err, args.blur_sig), g_range, linewidth=1, color='white')
#axs[0, 0].contour(plot_r_err, levels=[r_err_med-np.sqrt(args.tolerance)*(var_err), args.tolerance*r_err_med], extend='both', extent=extent, origin='upper', colors=['pink', 'orange'])
#axs[0, 0].contour(plot_mag, alpha=0.5, origin='upper', extend='both', extent=extent, colors='white')
im1 = axs[0].imshow(sp.ndimage.gaussian_filter(plot_mag, args.blur_sig), extent=extent, aspect=2)
divider1 = make_axes_locatable(axs[0])
cax1 = divider1.append_axes("right", size="20%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1, label='Magnetisation Strength')

axs[1].set_title('Reconstruction Error')
axs[1].set(ylim=[4, 2.5])
axs[1].set_xlabel('T')
axs[1].plot(T_c_2, g_range, linewidth=2, color='r')
axs[1].plot(sp.ndimage.gaussian_filter1d(gam_min_err, args.blur_sig), g_range, linewidth=1, color='white')
#axs[1, 0].contour(plot_mag, alpha=0.5, origin='upper', extend='both', extent=extent, colors='white')
#axs[1, 0].contour(plot_r_err, levels=[r_err_med-np.sqrt(args.tolerance)*(var_err), args.tolerance*r_err_med], extend='both', extent=extent, origin='upper', colors=['pink', 'orange'])
im2 = axs[1].imshow(sp.ndimage.gaussian_filter(plot_r_err, args.blur_sig), extent=extent, aspect=2)
divider2 = make_axes_locatable(axs[1])
cax2 = divider2.append_axes("right", size="20%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2, label='Magnitutde of Error')
"""
axs[0, 1].set_title('Susceptibility')
axs[0, 1].set(ylim=[4, 2.5])
axs[0, 1].set_xlabel('T')
axs[0, 1].plot(sp.ndimage.gaussian_filter1d(gam_max_sus, args.blur_sig), g_range, linewidth=1, color='cyan')
im3 = axs[0, 1].imshow(sp.ndimage.gaussian_filter(plot_sus, args.blur_sig), extent=extent, aspect='auto')
im3.set_clim(0, 0.003)
divider3 = make_axes_locatable(axs[0, 1])
cax3 = divider3.append_axes("right", size="20%", pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3, label='Magnitutde of Chi')

axs[1, 1].set_title('no_h_err')
axs[1, 1].set(ylim=[4, 2.5])
axs[1, 1].set_xlabel('T')
axs[1, 1].plot(T_c_2, g_range, linewidth=2, color='r')
im4 = axs[1, 1].imshow(sp.ndimage.gaussian_filter(plot_no_h_err, args.blur_sig), extent=extent, aspect='auto')
divider4 = make_axes_locatable(axs[1, 1])
cax4 = divider4.append_axes("right", size="20%", pad=0.05)
cbar4 = plt.colorbar(im4, cax=cax4, label='Magnitutde of Error')
"""
plt.tight_layout()
# Make space for title
#plt.subplots_adjust(top=0.9)
plt.show()
n = n[0]
cut_list = [int(n/4), int(n/3), int(3*n/5), int(7*n/8)]

fig1, ax = plt.subplots(2, 2, sharex=True)
ax[0, 0].set_title(gamma[cut_list[0]-1])
ax[0, 0].plot(T_range, plot_r_err[cut_list[0], :], label='Error')
ax2 = ax[0, 0].twinx()
ax22 = ax[0, 0].twinx()
ax2.plot(T_range, plot_sus[cut_list[0], :], color='r', label='Sus')
ax22.plot(T_range, plot_mag[cut_list[0], :], color='g', label='Mag')
ax[0, 0].set_yscale('log')
ax2.set_yscale('log')
ax22.set_yscale('log')

ax[1, 0].set_title(gamma[cut_list[1]-1])
ax[1, 0].plot(T_range, plot_r_err[cut_list[1], :], label='Error')
ax3 = ax[1, 0].twinx()
ax33 = ax[1, 0].twinx()
ax3.plot(T_range, plot_sus[cut_list[1], :], color='r', label='Sus')
ax33.plot(T_range, plot_mag[cut_list[1], :], color='g', label='Mag')
ax[1, 0].set_yscale('log')
ax3.set_yscale('log')
ax33.set_yscale('log')

ax[0, 1].set_title(gamma[cut_list[2]-1])
ax[0, 1].plot(T_range, plot_r_err[cut_list[2], :], label='Error')
ax4 = ax[0, 1].twinx()
ax44 = ax[0, 1].twinx()
ax4.plot(T_range, plot_sus[cut_list[2], :], color='r', label='Sus')
ax44.plot(T_range, plot_mag[cut_list[2], :], color='g', label='Mag')
ax[0, 1].set_yscale('log')
ax4.set_yscale('log')
ax44.set_yscale('log')

ax[1, 1].set_title(gamma[cut_list[3]-1])
ax[1, 1].plot(T_range, plot_r_err[cut_list[3], :], label='Error')
ax5 = ax[1, 1].twinx()
ax55 = ax[1, 1].twinx()
ax5.plot(T_range, plot_sus[cut_list[3], :], color='r', label='Sus')
ax55.plot(T_range, plot_mag[cut_list[3], :], color='g', label='Mag')
ax[1, 1].set_yscale('log')
ax5.set_yscale('log')
ax55.set_yscale('log')
plt.legend()
plt.tick_params(labelleft = False)
plt.show()
"""
plt.figure()
plt.imshow(max_sus)
plt.clim(0, 0.003)
plt.show()
"""

plt.figure()
plt.plot(g_range, (sp.ndimage.gaussian_filter1d(gam_max_sus, args.blur_sig)), color='g')
plt.plot(g_range, (sp.ndimage.gaussian_filter1d(gam_min_err, args.blur_sig)))
plt.show()

err_shift = np.subtract(gam_min_err, gam_max_sus)
mean_err_shift = np.mean(err_shift)
plt.figure()
plt.plot(g_range, sp.ndimage.gaussian_filter1d(err_shift, args.blur_sig))
#plt.plot(g_range, err_shift)
plt.hlines(0, 2.5, 5, linestyle='--', color='black')
plt.xlabel('gamma')
plt.ylabel('Error Shift')
plt.text(3.6,-1, 'Ordered Phase', color='green')
plt.text(3.6, 1, 'Disordered Phase', color='r')
plt.xlim(2.5,4)
plt.ylim(-1.5, 1.5)
plt.savefig("C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Images/" + str(indir) + "Errshift.png", dpi=500, format='png')
plt.show()
print('mean err shift:' + str(mean_err_shift))


plt.figure()
plt.title('Absolute Value of Magnetisation')
#axs[0, 0].set(ylim=[5, 2.4])
plt.xlabel('T')
plt.plot(T_c_2, g_range, linewidth=2, color='r', label='Analytic Solution')
plt.hlines(3, 0, 10, colors='orange', alpha=0.6, linestyles='--')
plt.ylabel('Gamma')
plt.ylim(4, 2.5)
plt.plot(sp.ndimage.gaussian_filter1d(gam_max_sus, args.blur_sig), g_range, linewidth=1, color='cyan', label='Maximum Susceptibility')
#plt.plot(sp.ndimage.gaussian_filter1d(gam_min_err, args.blur_sig), g_range, linewidth=1, color='white')
#axs[0, 0].contour(plot_r_err, levels=[r_err_med-np.sqrt(args.tolerance)*(var_err), args.tolerance*r_err_med], extend='both', extent=extent, origin='upper', colors=['pink', 'orange'])
#axs[0, 0].contour(plot_mag, alpha=0.5, origin='upper', extend='both', extent=extent, colors='white')
im7 = plt.imshow(sp.ndimage.gaussian_filter(plot_mag, args.blur_sig), extent=extent, aspect='auto')
#divider1 = make_axes_locatable(axs[0, 0])
#cax1 = divider1.append_axes("right", size="20%", pad=0.05)
cbar7 = plt.colorbar(im7, label='Magnetisation Strength')
plt.legend(loc=4)
plt.savefig("C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Images/" + str(indir) + ".png", dpi=500, format='png')
plt.show()


residual_T_c = np.subtract(gam_max_sus, T_c_2)
mean_res = np.mean(abs(residual_T_c))
plt.figure()
plt.scatter(g_range, residual_T_c, marker='x')
plt.xlabel('Gamma')
plt.ylabel('Residual Value')
#plt.text(3.2,-50, mean_res, color='black')
#plt.legend()
#plt.savefig("C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Images/" + str(indir) + str(mean_res) + ".png", dpi=500, format='png')
plt.show()
print(mean_res)


plt.figure()
plt.title('Absolute Value of Magnetisation')
plt.xlabel('T')
plt.plot(T_c_2, g_range, linewidth=2, color='r', label='Analytic Solution')
plt.hlines(3, 0, 10, colors='orange', alpha=0.6, linestyles='--')
plt.ylabel('Gamma')
plt.ylim(4, 2.5)
plt.plot(sp.ndimage.gaussian_filter1d(gam_max_sus, args.blur_sig), g_range, linewidth=1, color='cyan', label='Maximum Susceptibility')
plt.plot(sp.ndimage.gaussian_filter1d(gam_min_err, args.blur_sig), g_range, linewidth=1, color='white', label='Minimum Error')
#axs[0, 0].contour(plot_r_err, levels=[r_err_med-np.sqrt(args.tolerance)*(var_err), args.tolerance*r_err_med], extend='both', extent=extent, origin='upper', colors=['pink', 'orange'])
#axs[0, 0].contour(plot_mag, alpha=0.5, origin='upper', extend='both', extent=extent, colors='white')
im8 = plt.imshow(sp.ndimage.gaussian_filter(plot_mag, args.blur_sig), extent=extent, aspect='auto')
#divider1 = make_axes_locatable(axs[0, 0])
#cax1 = divider1.append_axes("right", size="20%", pad=0.05)
cbar8 = plt.colorbar(im8, label='Magnetisation Strength')
plt.legend(loc=4)
plt.savefig("C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Images/" + str(indir) + "magwerr.png", dpi=500, format='png')
plt.show()

plt.figure()
plt.title('Reconstruction Error')
plt.xlabel('T')
plt.plot(T_c_2, g_range, linewidth=2, color='r', label='Analytic Solution')
plt.hlines(3, 0, 10, colors='orange', alpha=0.6, linestyles='--')
plt.ylabel('Gamma')
plt.ylim(4, 2.5)
plt.plot(sp.ndimage.gaussian_filter1d(gam_max_sus, args.blur_sig), g_range, linewidth=1, color='cyan', label='Maximum Susceptibility')
plt.plot(sp.ndimage.gaussian_filter1d(gam_min_err, args.blur_sig), g_range, linewidth=1, color='white', label='Minimum Error')
#axs[0, 0].contour(plot_r_err, levels=[r_err_med-np.sqrt(args.tolerance)*(var_err), args.tolerance*r_err_med], extend='both', extent=extent, origin='upper', colors=['pink', 'orange'])
#axs[0, 0].contour(plot_mag, alpha=0.5, origin='upper', extend='both', extent=extent, colors='white')
im9 = plt.imshow(sp.ndimage.gaussian_filter(plot_r_err, args.blur_sig), extent=extent, aspect='auto')
#divider1 = make_axes_locatable(axs[0, 0])
#cax1 = divider1.append_axes("right", size="20%", pad=0.05)
cbar9 = plt.colorbar(im9, label='Error Value')
plt.legend(loc=4)
plt.savefig("C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Images/" + str(indir) + "err.png", dpi=500, format='png')
plt.show()