import sys
import numpy as np

from tqdm import tqdm
from os.path import join

import inference.analysis.new as analysis
from inference.io import Readhdf5_mc

run_dir = sys.argv[1]
input_file = join(run_dir, 'mc_output.hdf5')
with Readhdf5_mc(input_file, True) as f:
    trajs = f.read_many_datasets('configurations')
    md = f.get_metadata()
    temps = md['SweepParameterValues']
    sampling_frequency = md['CycleDumpFreq']

nTemps, nSamples, nSpins = trajs.shape
taus = np.empty(nTemps)
autocorrs = np.empty((nTemps, nSamples))

for c, traj in enumerate(tqdm(trajs)):
    observables = analysis.Correlations(traj)
    t, ac, tau = observables.scipy_autocorr(
        calc_correlation_time=True)
    autocorrs[c, :] = ac
    taus[c] = tau
    t_real = t * sampling_frequency

out_file = join(run_dir, 'AutoCorrelation.npz')
np.savez(
    out_file, T=temps, tau_alpha=taus, real_times=t_real, autocorrs=autocorrs)
