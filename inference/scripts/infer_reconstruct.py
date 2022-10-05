import os
import glob
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trajfile', nargs='?', required=True)
args = parser.parse_args()


def collect_model(row_file, model):
    npz_file = np.load(row_file)
    if model is None:
        N = npz_file['iN'][1]
        model = np.empty((N, N))

    row_index = npz_file['iN'][0]
    bias = npz_file['b']
    left_weights = npz_file['wl']
    right_weights = npz_file['wr']
    model[row_index, 0:row_index] = left_weights
    model[row_index, row_index+1:] = right_weights
    model[row_index, row_index] = bias
    return model


path, file = os.path.split(args.trajfile)
row_dir = os.path.join(path, "Rows-{}".format(file[:-5]))
row_files = np.array(glob.glob(os.path.join(row_dir, 'row*')))

model = None
for row_file in row_files:
    model = collect_model(row_file, model)
model = (model + model.T) * 0.5

with h5py.File(args.trajfile, 'a') as f:
    inferred_model_dataset = f.require_dataset(
        'InferredModel',
        shape=model.shape, dtype=model.dtype,
        compression='gzip')
    inferred_model_dataset[()] = model
