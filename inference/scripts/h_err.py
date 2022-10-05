import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import inference.analysis.planalysis as planalysis
from inference import tools

path = os.path.join('testSW_2', fname)
with h5py.File(path, 'r') as fin:
    modIn = fin['InputModel'][()]                   # reloading file as np array 
    modOut = fin['InferredModel'][()]
    md = dict(fin['configurations'].attrs.items())

fig, ax = plt.subplots(1, 2)
ax = ax.ravel()
ax[0].imshow(modIn)
ax[1].imshow(modOut)
plt.show()
fig, ax = plt.subplots(1, 2)
ax = ax.ravel()

J_in = tools.triu_flat(modIn)
h_in = np.diagonal(modIn)
J_out = tools.triu_flat(modOut)
h_out = np.diagonal(modOut)