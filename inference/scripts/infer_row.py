import os
import sys
import argparse
import h5py
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()

parser.add_argument('--trajfile', nargs='?', required=True)
parser.add_argument('--rowindex', nargs='?', type=int, required=True)
args = parser.parse_args()


with h5py.File(args.trajfile, 'r') as f:
    config_dset = f['configurations']
    if "eq_cycles" in dict(config_dset.attrs.items()):
        discard = int(
            config_dset.attrs["eq_cycles"] /
            config_dset.attrs["cycle_dumpfreq"])
        traj = config_dset[discard:, :]
    else:
        traj = config_dset[()]

print(traj.shape)


row_index = args.rowindex
nSamples, nFeatures = traj.shape

if args.rowindex > nFeatures - 1:
    sys.exit(1)
# this should be something else!
# model_sklearn = np.empty((nFeatures, nFeatures))

# for row_index in range(0, nFeatures):
X = np.delete(traj, row_index, 1)
y = traj[:, row_index]  # target
log_reg = LogisticRegression(
    penalty='none',
    # C=0.1,
    # random_state=0,
    solver='lbfgs',
    max_iter=200)
log_reg.fit(X, y)
weights = log_reg.coef_[0] / 2  # factor of 2 from equations!
bias = log_reg.intercept_[0] / 2
left_weights = weights[0:row_index]
right_weights = weights[row_index:]

path, file = os.path.split(args.trajfile)

rows_outpath = os.path.join(path, "Rows-{}".format(file[:-5]))
Path(rows_outpath).mkdir(parents=True, exist_ok=True)

row_outpath = os.path.join(rows_outpath, "row{}.npz".format(args.rowindex))
np.savez(
    row_outpath,
    iN=[row_index, nFeatures],
    b=bias, wl=left_weights,
    wr=right_weights)

# plt.imshow(model_sklearn)
# plt.show()