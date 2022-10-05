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
import networkx as nx
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable


def inf_plots(folder, fname):
    path = os.path.join(folder, fname)
    with h5py.File(path, 'r') as fin:
        modIn = fin['InputModel'][()]                   # reloading file as np array 
        modOut = fin['InferredModel'][()]
        md = dict(fin['configurations'].attrs.items())
        md_in = dict(fin['InputModel'].attrs.items())
        T = md.get('T')
        config_traj = np.array(fin.get('configurations'))
        #sus = (1/T)*np.array(fin.get('configurations')).var()
    #Now = datetime.datetime.now()
    #now = Now.strftime('%f')
    fig1, ax1 = plt.subplots(1, 2)
    #ax1 = ax1.ravel()
    im0 = ax1[0].imshow(modIn)
    im1 = ax1[1].imshow(modOut)
    divider0 = make_axes_locatable(ax1[0])
    divider1 = make_axes_locatable(ax1[1])
    cax0 = divider0.append_axes("right", size="10%", pad=0.05)
    cax1 = divider1.append_axes("right", size="10%", pad=0.05)
    cbar0 = plt.colorbar(im0, cax=cax0, label='J_in Strength')
    cbar1 = plt.colorbar(im1, cax=cax1, label='J_inf Strength')
    plt.tight_layout()
    #plt.title(fname)
    plt.savefig("C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Images/J_comp/" + str(fname) + "J_comp.png", dpi=500, format='png')
    plt.close()
    
    N = md.get('N')
    T = md.get('T')
    h = md.get('h')
    gamma = md.get('gamma')
    n = md_in.get('n')
    model = md_in.get('model')
    
    # Plot DegDist and find k_0 for graph
    graph = nx.from_numpy_array(modIn)
    Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    degree_sequence = sorted((d for n, d in Gcc.degree()), reverse=True)
    k_0 = min(degree_sequence)
    # Plot
    
    fig, ax = plt.subplots(1, 2)
    #ax = ax.ravel()
    ax[0].bar(*np.unique(degree_sequence, return_counts=True))
    ax[0].set_title("Degree Histogram")
    ax[0].set_xlabel("Degree")
    ax[0].set_ylabel("Frequency")
    im2 = ax[1].imshow(modIn)
    #plt.title(fname)
    cbar2 = plt.colorbar(im2, label='J_in Strength')
    plt.savefig("C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Images/hist/" + str(fname) + "hist.png", dpi=500, format='png')
    
    plt.close(fig)
    
    magz = []
    config_len = len(config_traj[:,1])
    _mag = 0
    _mag = np.abs(np.sum(config_traj, axis=1))/N
    magz.append(_mag)
        
    mag = np.sum(magz)/config_len
    sus = np.var(magz)/T
        
    J_in = modIn
    h_in = np.diagonal(modIn)
    J_out = modOut
    h_out = np.diagonal(modOut)

    r_err = np.sqrt(np.sum((np.subtract(J_in, J_out))**2)/np.sum(J_in)**2)
    h_err = np.sqrt(np.sum((np.subtract(h_in, h_out))**2)/np.sum(J_in)**2)
    no_h_err = r_err - h_err
    
    return (r_err, mag, sus, N, T, h, gamma, n, no_h_err, model, k_0)

parser = argparse.ArgumentParser()
parser.add_argument('--indir', nargs='?', required=True)
parser.add_argument('--outdir', nargs='?', required=True)
parser.add_argument('--nJobs', type=int, required=True)
args = parser.parse_args()

# set parsed inputs as vars:
files_folder = args.indir
nJobs = args.nJobs
# List path to and file names in tuple:
files = data_namer.state_point_lister(files_folder)
# Fn to iterate over calculator
output = [inf_plots(files[1], files[0][i]) for i in range(len(files[0]))]
t0 = perf_counter() 
values = Parallel(n_jobs=nJobs)(delayed(inf_plots)(files[1], files[0][i]) for i in tqdm(range(len(files[0]))))
get_reusable_executor().shutdown(wait=True)
t1 = perf_counter()
print('Time taken: {}s'.format(t1 - t0))

fields = ['r_err', 'mag', 'sus', 'N', 'T', 'h', 'gamma', 'n', 'no_h_err', 'model', 'k_0']
with open(str(args.outdir) + '.csv', 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile, delimiter=':') 
    # writing the fields 
    csvwriter.writerow(fields)  
    # writing the data rows 
    csvwriter.writerows(values)