import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import networkx as nx
# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.linear_model import LogisticRegression
# from joblib import Parallel, delayed

import inference.core.oscar_utils as utils
from inference.sweep import MonteCarlo2

h5py.get_config().track_order = True


def param_range(parg, nStatepoints):
    if len(parg) == 1:
        pRange = parg
    elif len(parg) == 2:
        pRange = np.linspace(np.min(parg), np.max(parg), nStatepoints + 1)
    else:
        print('its not guuud')
    return pRange


def genSW(parallel_arguments):
    # print(parallel_arguments)
    T = parallel_arguments[0]
    h = parallel_arguments[1]
    prob = parallel_arguments[2]
    k = parallel_arguments[3]
    modlabel = parallel_arguments[4]
    N = parallel_arguments[5]
    outdir = parallel_arguments[6]
    # print(N, T, h, prob, k)
    md = {
        "T": T,
        "h": h,
        "prob": prob,
        "k": k,
        "N": N,
        "outdir": outdir,
        "model": "SW"
        }
    model, graph = utils.SW_matrix(N, k, prob, h, T)              # make array for specific model u want to use - should be a straight swap e.g o_utils.SM_matrix(40, 2, 1, 0, 1) 
    plt.imshow(model)
    plt.show()
    nx.draw_kamada_kawai(graph, **{'node_size': 20, 'width': 0.2})
    plt.show()
    h5py_fname = os.path.join(outdir, modlabel)
    h5py_fname = h5py_fname + '.hdf5'
    with h5py.File(h5py_fname, "a") as f:
        # data_group = f.require_group(args.modlabel)
        # print('Saving InputModel: {}'.format(modlabel))
        ds = f.create_dataset(
            "InputModel", data=model, compression="gzip")
        for key, val in md.items():
            ds.attrs[key] = val
    return h5py_fname

def genSF(parallel_arguments):
    # print(parallel_arguments)
    T = parallel_arguments[0]
    h = parallel_arguments[1]
    gamma = parallel_arguments[2]
    modlabel = parallel_arguments[3]
    N = parallel_arguments[4]
    outdir = parallel_arguments[5]
    n = parallel_arguments[6]
    # print(N, T, h, J0, Jstd)
    md = {
        "T": T,
        "h": h,
        "gamma": gamma,
        "N": N,
        "outdir": outdir,
        "model": "SF",
        "n": n
        }
    model, graph = utils.SF_matrix(N, gamma, h, T)      # make array for specific model u want to use - should be a straight swap e.g o_utils.SM_matrix(40, 2, 1, 0, 1) 
    plt.imshow(model)
    plt.show()
    nx.draw_kamada_kawai(graph, **{'node_size': 1, 'width': 0.15})
    #ax = plt.axes()
    #ax.patch.set_facecolor('black')
    #ax.patch.set_alpha(0.1)
    plt.show()
    h5py_fname = os.path.join(outdir, modlabel)
    h5py_fname = h5py_fname + '.hdf5'
    with h5py.File(h5py_fname, "a") as f:
        # data_group = f.require_group(args.modlabel)
        # print('Saving InputModel: {}'.format(modlabel))
        ds = f.create_dataset(
            "InputModel", data=model, compression="gzip")
        for key, val in md.items():
            ds.attrs[key] = val
    return h5py_fname


def mccSW(h5py_fname, mcc_args):
    with h5py.File(h5py_fname, 'r') as fin:                     # set metadata
        model_dataset = fin['InputModel']
        model = model_dataset[:]
        modparams = {}
        # print(model_dataset.attrs['modparams'])
        modparams['T'] = model_dataset.attrs['T']
        modparams['h'] = model_dataset.attrs['h']
        modparams['prob'] = model_dataset.attrs['prob']
        modparams['k'] = model_dataset.attrs['k']

    ising_sim = MonteCarlo2(model, modparams, h5py_fname)        # run sim
    # print('----')
    ising_sim.setHyperParameters(
        eq_cycles=mcc_args[0],
        prod_cycles=mcc_args[1],
        cycle_dumpfreq=mcc_args[2]
        )
    ising_sim.run()
    
    
def mccSF(h5py_fname, mcc_args):
    with h5py.File(h5py_fname, 'r') as fin:                      # set metadata
        model_dataset = fin['InputModel']
        model = model_dataset[:]
        modparams = {}
        # print(model_dataset.attrs['modparams'])
        modparams['T'] = model_dataset.attrs['T']
        modparams['h'] = model_dataset.attrs['h']
        modparams['gamma'] = model_dataset.attrs['gamma']

    ising_sim = MonteCarlo2(model, modparams, h5py_fname)        # run sim
    # print('----')
    ising_sim.setHyperParameters(
        eq_cycles=mcc_args[0],
        prod_cycles=mcc_args[1],
        cycle_dumpfreq=mcc_args[2]
        )
    ising_sim.run()


def infer(h5py_fname):
    with h5py.File(h5py_fname, 'r') as f:
        config_dset = f['configurations']
        if "eq_cycles" in dict(config_dset.attrs.items()):
            discard = int(
                config_dset.attrs["eq_cycles"] /
                config_dset.attrs["cycle_dumpfreq"])
            traj = config_dset[discard:, :]
        else:
            traj = config_dset[()]
    # print(traj.shape)
    nSamples, nFeatures = traj.shape
    model_inf = np.zeros((nFeatures, nFeatures))
    # Parallel(n_jobs=-1, max_nbytes=None)(
    #     delayed(logRegLoop_inner)(traj, model_inf, row_index)
    #     for row_index in range(0, nFeatures)
    # )
    for row_index in range(0, nFeatures):
        logRegLoop_inner(traj, model_inf, row_index)
    #     # X = np.delete(traj, row_index, 1)
    #     # y = traj[:, row_index]  # target
    #     # log_reg = LogisticRegression(
    #     #     penalty='none',
    #     #     # C=0.1,
    #     #     # random_state=0,
    #     #     solver='lbfgs',
    #     #     max_iter=200)
    #     # try:
    #     #     log_reg.fit(X, y)
    #     #     weights = log_reg.coef_[0] / 2  # factor of 2 from equations!
    #     #     bias = log_reg.intercept_[0] / 2
    #     #     left_weights = weights[0:row_index]
    #     #     right_weights = weights[row_index:]
    #     #     model_inf[row_index, 0:row_index] = left_weights
    #     #     model_inf[row_index, row_index+1:] = right_weights
    #     #     model_inf[row_index, row_index] = bias
    #     # except ValueError:
    #     #     model_inf[row_index] = 20 * np.ones_like(model_inf[row_index])

    # model_inf = (model_inf + model_inf.T) * 0.5
    with h5py.File(h5py_fname, 'a') as f:
        inferred_model_dataset = f.require_dataset(
            'InferredModel',
            shape=model_inf.shape, dtype=model_inf.dtype,
            compression='gzip')
        inferred_model_dataset[()] = model_inf


def logRegLoop_inner(traj, model_inf, row_index):
    X = np.delete(traj, row_index, 1)
    y = traj[:, row_index]  # target
    log_reg = LogisticRegression(
        penalty='none',
        # C=0.1,
        # random_state=0,
        solver='lbfgs',
        max_iter=1000)
    try:
        log_reg.fit(X, y)               # X is all the spins, bar that row,, y is the spins from that row
        weights = log_reg.coef_[0] / 2  # factor of 2 from equations!  weights = w in doc and Jir in paper -- size = N-1
        bias = log_reg.intercept_[0] / 2  # intercept is the h_r and c in doc -- size = 1
        left_weights = weights[0:row_index]     #left of diag
        right_weights = weights[row_index:]
        model_inf[row_index, 0:row_index] = left_weights
        model_inf[row_index, row_index+1:] = right_weights
        model_inf[row_index, row_index] = bias
    except ValueError:
        model_inf[row_index] = 20 * np.ones_like(model_inf[row_index])      # may want to try diff value than 20 
