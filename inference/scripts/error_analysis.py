import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob
import argparse
import h5py
from os.path import join

import inference.analysis.new as analysis
import inference.analysis.planalysis as planalysis
from inference import tools


# parser = argparse.ArgumentParser()
# parser.add_argument('--infile', nargs='*', required=True)

# args = parser.parse_args()
plt.style.use('~/Devel/styles/custom.mplstyle')
# infile = args.infile
infile = './N200_1/T_1.325-h_0-J_0-Jstd_1.hdf5'
# infile = './N200_1/T_1.325-h_0-J_0-Jstd_1.hdf5'
# infile = './N400_1/T_1.325-h_0-J_0-Jstd_1.hdf5'

infile = '/Users/mk14423/Devel/fmri2/executables/inftest/T_1.475-h_0-J_0-Jstd_1.hdf5'
print(infile)
# infile = infile[0]
with h5py.File(infile, 'r') as f:
    traj_dataset = f['configurations']
    metadata = dict(traj_dataset.attrs.items())
    true_model = f['InputModel'][()]
    inferred_model = f['InferredModel'][()]

# inferred_model[inferred_model != 0] = 1
# zero_indicies = np.argwhere(inferred_model == 0)
# for qestionable_index in zero_indicies:
#     print(qestionable_index)

# plt.imshow(inferred_model)
# plt.show()
# print(inferred_model[0, 1], inferred_model[1, 0])
ErrorPipe = planalysis.ErrorAnalysis(true_model, inferred_model)
ErrorPipe.overview()
ErrorPipe.histograms()
error = planalysis.recon_error_nguyen(
                true_model, inferred_model)
print(error)
infile = '/Users/mk14423/Devel/fmri2/executables/inftest/T_1.48-h_0-J_0-Jstd_1.hdf5'

# infile = './N400_1/T_1.475-h_0-J_0-Jstd_1.hdf5'
print(infile)
# infile = infile[0]
with h5py.File(infile, 'r') as f:
    traj_dataset = f['configurations']
    metadata = dict(traj_dataset.attrs.items())
    true_model = f['InputModel'][()]
    inferred_model = f['InferredModel'][()]

ErrorPipe = planalysis.ErrorAnalysis(true_model, inferred_model)
ErrorPipe.overview()
ErrorPipe.histograms()
error = planalysis.recon_error_nguyen(
                true_model, inferred_model)
print(error)