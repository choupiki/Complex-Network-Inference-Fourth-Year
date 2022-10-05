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
if df.model[5] == 'CM' or df.model[5] == '2D Ising Model':
    df = df.sort_values(by=['T'])
    df
    # Read results and facility params from dataframe
    T = np.array(df.T)
    T = T[:][4]
    n = np.array(df.n)
    r_err = df.r_err
    mag = df.mag
    sus = df.sus
    N = df.N
    N = N[1]
    no_h_err = df.no_h_err
    model = df.model
    print(df, 'n\n\n T:', T, 'n\n\n', mag, 'n\n\n', r_err, 'n\n\n', sus, 'n\n\n', model)
    print(T[-1])
    T_range = whelp.param_range([T[0], T[-1]], n[0])
    print(T_range)
    r_err_med = statistics.median(r_err)
    no_h_err_med = statistics.median(no_h_err)
    print('r_err median: \n\n', r_err_med)
    print('r_err median: \n\n', no_h_err_med)

    plot_mag = np.zeros((len(T_range)))
    plot_r_err = np.zeros((len(T_range)))
    plot_sus = np.zeros((len(T_range)))
    plot_no_h_err = np.zeros((len(T_range)))
    for i in range(len(T_range)):
        plot_mag[i] = mag.iloc[i]
        plot_sus[i] = sus.iloc[i]
        if r_err.iloc[i] > args.tolerance*r_err_med:
            plot_r_err[i] = args.tolerance*r_err_med
            plot_no_h_err[i] = args.tolerance*no_h_err_med
        else:     
            plot_r_err[i] = r_err.iloc[i]
            plot_no_h_err[i] = no_h_err.iloc[i]

    T_min_r = np.argmin(plot_r_err)
    T_max_r = np.argmax(plot_r_err)
    T_max_sus = np.argmax(plot_sus)


    # Plotting
    fig, axs = plt.subplots(2, 2, sharey=False)
    fig.suptitle(model[0], fontsize=14)  
    """
    T_c = []
    for i in range(len(g_range)):
        if g_range[i] > 3.005:
            T_c.append(-2/(np.log((4-g_range[i])/(g_range[i]-2))))
        else:
            T_c.append(7)
    """        
    
    axs[0, 0].set_title('Absolute Value of Magnetisation')
    axs[0, 0].axvline(T_range[T_min_r]/N, linestyle='--')
    axs[0, 0].axvline(T_range[T_max_r]/N, linestyle='--', color='r')
    axs[0, 0].axvline(T_range[T_max_sus]/N, linestyle='--', color='g')
    axs[0, 0].set_xlabel('T')
    axs[0, 0].plot(T_range/N, plot_mag)
    axs[0, 0].set_ylabel('Mag', fontweight='bold')

    axs[1, 0].set_title('Reconstruction Error')
    #axs[1, 0].set(xlim=[0,4])
    axs[1, 0].axvline(T_range[T_min_r]/N, linestyle='--')
    axs[1, 0].axvline(T_range[T_max_r]/N, linestyle='--', color='r')
    axs[1, 0].axvline(T_range[T_max_sus]/N, linestyle='--', color='g')
    axs[1, 0].set_xlabel('T')
    axs[1, 0].plot(T_range/N, plot_r_err)
    
    axs[0, 1].set_title('Susceptibility')
    axs[0, 1].set_xlabel('T')
    axs[0, 1].axvline(T_range[T_min_r]/N, linestyle='--')
    axs[0, 1].axvline(T_range[T_max_r]/N, linestyle='--', color='r')
    axs[0, 1].axvline(T_range[T_max_sus]/N, linestyle='--', color='g')
    axs[0, 1].plot(T_range/N, plot_sus)
    ax2 = axs[1, 0].twinx()
    axs[1, 0].set(yscale='log')
    ax2.set(yscale='log')
    ax2.plot(T_range/N, plot_sus, color='g')

    axs[1, 1].set_title('no_h_err')
    axs[1, 1].set_xlabel('T')
    axs[1, 1].axvline(T_range[T_max_sus]/N, linestyle='--', color='g')
    axs[1, 1].plot(T_range/N, plot_no_h_err)
    plt.tight_layout()
    # Make space for title
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    plt.figure()
    plt.title('Absolute Value of Magnetisation')
    plt.xlabel('T')
    plt.plot(T_range, plot_mag)
    plt.ylabel('Mag', fontweight='bold')
    plt.show()
    """
    fig2, axs2 = plt.subplots(2, 1, sharex=True)
    axs2[0].set_title('Absolute Value of Magnetisation')
    axs2[0].set(xlim=[0,4])
    axs2[0].axvline(T_range[T_min_r]/N, linestyle='--', label='min err')
    axs2[0].axvline(T_range[T_max_r]/N, linestyle='--', color='r', label='max err')
    axs2[0].axvline(T_range[T_max_sus]/N, linestyle='--', color='g', label='max sus')
    #axs2[0].set_xlabel('T')
    axs2[0].plot(T_range/N, plot_mag)
    axs2[0].set_ylabel('Mag', fontweight='bold')
"""
    fig2, axs2 = plt.subplots(1, 1)
    axs2.set_title('Reconstruction Error')
    #axs2.set(xlim=[0.25,2])
    axs2.axvline(T_range[T_min_r]/N, linestyle='--', label='min err')
    axs2.axvline(T_range[T_max_r]/N, linestyle='--', color='r', label='max err')
    axs2.axvline(T_range[T_max_sus]/N, linestyle='--', color='g', label='max sus')
    axs2.set_xlabel('T')
    #axs2.plot(T_range/N, plot_r_err, label='error')
    ax3 = axs2.twinx()
    ax4 = axs2.twinx()
    axs2.set(yscale='log')
    ax3.set(yscale='log')
    ax4.set(yscale='log')
    #ax3.set_label('Susceptibility')
    ax3.plot(T_range/N, plot_sus, color='g', label='Susceptibility')
    ax4.plot(T_range/N, plot_mag, color='r', alpha=0.5, label='Magnetisation')
    axs2.set(yticklabels=[])
    ax3.set_yticklabels([]) 
    ax4.set_yticklabels([]) 
    ax4.tick_params(labelright='off')
    axs2.tick_params(right=False)
    #ax4.set_label('Magnetisation')
    ax3.legend()
    ax4.legend(loc='center right')
    plt.show()

    fig3, axs3 = plt.subplots(1, 1)
    axs3.set_title(str(df.model[5]) + ' Phase Transition')
    #axs2.set(xlim=[0.25,2])
    #axs2.axvline(T_range[T_min_r]/N, linestyle='--', label='min err')
    #axs2.axvline(T_range[T_max_r]/N, linestyle='--', color='r', label='max err')
    axs3.axvline(T_range[T_max_sus]/N, linestyle='--', color='g', label='max sus')
    axs3.set_xlabel('T/N')
    #axs2.plot(T_range/N, plot_r_err, label='error')
    ax7 = axs3.twinx()
    ax8 = axs3.twinx()
    axs3.set(yscale='log')
    ax7.set(yscale='log')
    #ax8.set(yscale='log')
    #ax3.set_label('Susceptibility')
    ax7.plot(T_range/N, plot_sus, color='g', label='Susceptibility')
    ax8.plot(T_range/N, plot_mag, color='blue', label='Magnetisation')
    axs3.set(yticklabels=[])
    ax7.set_yticklabels([]) 
    ax8.set_yticklabels([]) 
    ax8.tick_params(labelright='off')
    axs3.tick_params(right=False)
    #ax4.set_label('Magnetisation')
    #ax7.legend(loc=0)
    #ax8.legend(loc=2)
    #plt.xlim(0, 2)
    fig3.legend()
    plt.savefig("C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Images/" + str(indir) + "PT.png", dpi=500, format='png')
    plt.show()

# Scale-Free Plots
if df.model[5] == 'SF':
    df = df.sort_values(by=['T', 'gamma'])
    df
    # Read results and facility params from dataframe
    T = np.array(df.T)
    T = T[:][4]
    n = np.array(df.n)
    r_err = df.r_err
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
    print('r_err median: \n\n', r_err_med)
    print('r_err median: \n\n', no_h_err_med)

    plot_mag = np.zeros((len(T_range), len(g_range)))
    plot_r_err = np.zeros((len(T_range), len(g_range)))
    plot_sus = np.zeros((len(T_range), len(g_range)))
    plot_no_h_err = np.zeros((len(T_range), len(g_range)))
    for i in range(len(T_range)):
        for ii in range(len(g_range)):
            j = i*len(g_range) + ii
            plot_mag[i, ii] = mag.iloc[j]
            plot_sus[i, ii] = sus.iloc[j]
            if r_err.iloc[j] > args.tolerance*r_err_med:
                plot_r_err[i, ii] = r_err_med
                plot_no_h_err[i, ii] = no_h_err_med
            else:     
                plot_r_err[i, ii] = r_err.iloc[j]
                plot_no_h_err[i, ii] = no_h_err.iloc[j]

    # Plotting
    extent = [T[0] , T[-1], gamma[-1], gamma[0]]
    fig, axs = plt.subplots(2, 2, sharey=True)
    if hasattr(df, 'model') == False:
        fig.suptitle('Scale Free Results', fontsize=14)
    if hasattr(df, 'model') == True:
        fig.suptitle(df.model[0], fontsize=14)
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
    im1 = axs[0, 0].imshow(sp.ndimage.gaussian_filter(plot_mag, args.blur_sig), extent=extent, aspect='auto')
    divider1 = make_axes_locatable(axs[0, 0])
    cax1 = divider1.append_axes("right", size="20%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, label='Magnetisation Strength')

    axs[1, 0].set_title('Reconstruction Error')
    axs[1, 0].set_xlabel('T')
    axs[1, 0].plot(T_c, g_range, linewidth=2, color='r')
    im2 = axs[1, 0].imshow(sp.ndimage.gaussian_filter(plot_r_err, args.blur_sig), extent=extent, aspect='auto')
    divider2 = make_axes_locatable(axs[1, 0])
    cax2 = divider2.append_axes("right", size="20%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, label='Magnitutde of Error')

    axs[0, 1].set_title('Susceptibility')
    axs[0, 1].set_xlabel('T')
    im3 = axs[0, 1].imshow(sp.ndimage.gaussian_filter(plot_sus, args.blur_sig), extent=extent, aspect='auto')
    divider3 = make_axes_locatable(axs[0, 1])
    cax3 = divider3.append_axes("right", size="20%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3, label='Magnitutde of Chi')

    axs[1, 1].set_title('no_h_err')
    axs[1, 1].set_xlabel('T')
    axs[1, 1].plot(T_c, g_range, linewidth=2, color='r')
    im4 = axs[1, 1].imshow(sp.ndimage.gaussian_filter(plot_no_h_err, args.blur_sig), extent=extent, aspect='auto')
    divider4 = make_axes_locatable(axs[1, 1])
    cax4 = divider4.append_axes("right", size="20%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4, label='Magnitutde of Error')

    plt.tight_layout()
    # Make space for title
    plt.subplots_adjust(top=0.9)
    plt.show()

else: 
    print('Unrecognised Network Type')


