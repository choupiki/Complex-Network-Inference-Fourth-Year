import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import inference.analysis.planalysis as planalysis
from inference import tools


def nicer_hist(ax, data, nbins=50, **kwargs):
    weights = np.ones_like(data) / len(data)
    hist, bin_edges = np.histogram(data, bins=nbins, weights=weights)
    # print(hist.shape, bin_edges.shape)
    bin_centres = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    ax.plot(bin_centres, hist)
    # xlbl = r'${}$'
    ax.set(xlabel=r'$x$', ylabel=r'$P(x)$')
    # ax.set_ylim(bottom=0)
    # ax.hist(
    #         data, bins=nbins,
    #         weights=np.ones_like(data) / len(data), alpha=0.5)


fname = 'T_1.000-h_0.000-J_1.000-Jstd_1.000.hdf5'
path = os.path.join('test_tf', fname)
with h5py.File(path, 'r') as fin:
    modIn = fin['InputModel'][()]
    modOut = fin['InferredModel'][()]

fig, ax = plt.subplots(1, 2)
ax = ax.ravel()

J_in = tools.triu_flat(modIn)
h_in = np.diagonal(modIn)
J_out = tools.triu_flat(modOut)
h_out = np.diagonal(modOut)

nicer_hist(ax[0], J_in)
nicer_hist(ax[0], J_out)
nicer_hist(ax[0], J_in - J_out)

nicer_hist(ax[1], h_out)
nicer_hist(ax[1], h_in)
nicer_hist(ax[1], h_in - h_out)

print(np.mean(J_in), np.mean(J_out), np.mean(J_in - J_out))
print(np.std(J_in), np.std(J_out), np.std(J_in - J_out))
print(planalysis.recon_error_nguyen(modIn, modOut))
plt.show()

paramsIn = tools.triu_flat(modIn, k=0)
paramsOut = tools.triu_flat(modOut, k=0)
plt.plot(paramsIn, paramsOut, ls='none', marker='.')
plt.show()
