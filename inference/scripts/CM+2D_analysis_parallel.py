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

def inf_plots(folder, fname):
    path = os.path.join(folder, fname)
    with h5py.File(path, 'r') as fin:
        modIn = fin['InputModel'][()]                   # reloading file as np array 
        modOut = fin['InferredModel'][()]
        md = dict(fin['configurations'].attrs.items())
        md_in = dict(fin['InputModel'].attrs.items())
        T = md.get('T')
        mag = np.abs(np.array(fin.get('configurations')).mean())
        sus = (1/T)*np.var(np.array(fin.get('configurations')))
    """
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(modIn)
    ax[1].imshow(modOut)
    plt.title(fname[i])
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    plt.show()
    """
    J_in = modIn
    h_in = np.diagonal(modIn)
    J_out = modOut
    h_out = np.diagonal(modOut)

    r_err = np.sqrt(np.sum((np.subtract(J_in, J_out))**2)/np.sum(J_in)**2)
    h_err = np.sqrt(np.sum((np.subtract(h_in, h_out))**2)/np.sum(J_in)**2)
    no_h_err = r_err - h_err
    
    N = md.get('N')
    T = md.get('T')
    h = md.get('h')
    model = md_in.get('model')
    n = md_in.get('n')

    return (r_err, mag, sus, N, T, h, n, no_h_err, model)

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

fields = ['r_err', 'mag', 'sus', 'N', 'T', 'h', 'n', 'no_h_err', 'model']
with open(str(args.outdir) + '.csv', 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile, delimiter=':') 
    # writing the fields 
    csvwriter.writerow(fields)  
    # writing the data rows 
    csvwriter.writerows(values)