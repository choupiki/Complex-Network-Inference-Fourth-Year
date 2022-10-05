import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import data_namer as dn
import inference.analysis.planalysis as planalysis
from inference import tools
import argparse

def nicer_hist(ax, data, nbins=50, **kwargs):
    weights = np.ones_like(data) / len(data)
    hist, bin_edges = np.histogram(data, bins=nbins, weights=weights)
    bin_centres = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    ax.plot(bin_centres, hist)
    ax.set(xlabel=r'$x$', ylabel=r'$P(x)$')

parser = argparse.ArgumentParser()
parser.add_argument('--indir', nargs='?', required=True)
args = parser.parse_args()

fname, path = dn.state_point_lister(args.indir)
# fname = 'T_1.000-h_0.000-J_1.000-Jstd_1.000.hdf5'
# fname = 'T_1.000-h_0.000-J_0.500-Jstd_1.000.hdf5'
# fname = 'T_1.200-h_0.000-J_0.750-Jstd_1.000.hdf5'
# fname = 'T_1.400-h_0.000-J_0.750-Jstd_1.000.hdf5'
# fname = 'T_1.400-h_0.000-J_0.500-Jstd_1.000.hdf5'

# fname = 'T_1.000-h_0.000-J_0.500-Jstd_0.800.hdf5'
# fname = 'T_1.000-h_0.000-J_0.750-Jstd_1.000.hdf5'
# fname = 'T_1.000-h_0.000-J_1.000-Jstd_1.200.hdf5'
def model_plot(indir, fname):
    path = os.path.join(indir, fname)
    with h5py.File(path, 'r') as fin:
        modIn = fin['InputModel'][()]                   # reloading file as np array 
        modOut = fin['InferredModel'][()]
        md = dict(fin['configurations'].attrs.items())

    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(modIn)
    ax[1].imshow(modOut)
    plt.show()
    
    plt.imshow(modIn)
    plt.show()

    J_in = tools.triu_flat(modIn)
    h_in = np.diagonal(modIn)
    J_out = tools.triu_flat(modOut)
    
    print(np.mean(J_in), np.mean(J_out), np.mean(J_in - J_out))
    print(np.std(J_in), np.std(J_out), np.std(J_in - J_out))
    print(planalysis.recon_error_nguyen(modIn, modOut), '\n\n\n\n')

    return

for i in range(len(fname)):
    if i//100 == 0:
        model_plot(args.indir, fname[i])
    else:
        pass
# Goodbye x
