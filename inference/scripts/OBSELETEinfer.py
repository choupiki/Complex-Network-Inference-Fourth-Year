
import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

# not using this anymore!!!
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()

parser.add_argument('--trajfile', nargs='?', required=True)
args = parser.parse_args()

# aha! I do need to make sure this only uses the production runs!
# so need to use the metadata to get the cut!
with h5py.File(args.trajfile, 'r') as f:
    if "eq_cycles" in dict(f.attrs.items()):
        discard = f.attrs["eq_cycles"]
        traj = f['configurations'][discard:, :]
    else:
        traj = f['configurations'][()]
print(traj.shape)

nSamples, nFeatures = traj.shape
model_sklearn = np.empty((nFeatures, nFeatures))

for row_index in range(0, nFeatures):
    X = np.delete(traj, row_index, 1)
    y = traj[:, row_index]  # target
    log_reg = LogisticRegression(
        penalty='none',
        # C=0.1,
        # random_state=0,
        solver='lbfgs',
        max_iter=200)
    log_reg.fit(X, y)
    weights = log_reg.coef_[0]
    print(row_index, y.shape)
    bias = log_reg.intercept_[0]
    left_weights = weights[0:row_index]
    right_weights = weights[row_index:]
    model_sklearn[row_index, 0:row_index] = left_weights
    model_sklearn[row_index, row_index+1:] = right_weights
    model_sklearn[row_index, row_index] = bias


model_sklearn = (model_sklearn + model_sklearn.T) / 2
model_sklearn /= 2
model = model_sklearn
with h5py.File(args.trajfile, 'a') as f:
    inferred_model_dataset = f.require_dataset(
        'InferredModel',
        shape=model.shape, dtype=model.dtype,
        compression='gzip')
    inferred_model_dataset[()] = model

# plt.imshow(model_sklearn)
# plt.show()
