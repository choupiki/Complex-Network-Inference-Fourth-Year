import numpy as np
import h5py
import matplotlib.pyplot as plt
import inference.preprocess as pp

conditions = ['noMM', 'MM', 'all']
# conditions = ['MM']
for condition in conditions:
    _, z, spin_configs = pp.load(condition)
    sampled_days, samples_per_day, N = spin_configs.shape
    spin_configs = spin_configs.reshape(sampled_days * samples_per_day, N)
    samples_tot, N = spin_configs.shape
    with h5py.File(condition + '.hdf5', 'w') as f:

        config_ds = f.require_dataset(
            "configurations",
            shape=(samples_tot, N),
            dtype=spin_configs.dtype,
            compression="gzip")
        config_ds[()] = spin_configs
# somehow zip the days together! i.e. take the stack and append.