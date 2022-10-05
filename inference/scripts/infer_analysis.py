import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score
import inference.tools as tools

#plt.style.use('~/Devel/styles/custom.mplstyle')
parser = argparse.ArgumentParser()
parser.add_argument('--trajfile', nargs='?', required=True)

args = parser.parse_args()

with h5py.File(args.trajfile, 'r') as fin:
    trueModel = fin["InputModel"][()]
    inferredModel = fin["InferredModel"][()]

true_params = tools.triu_flat(trueModel)
inferred_params = tools.triu_flat(inferredModel)
r2_val = r2_score(true_params, inferred_params)
# print(true_params)
# print(inferred_params)

fig = plt.figure()
gs = fig.add_gridspec(2, 2)
f_ax0 = fig.add_subplot(gs[0, 0])
f_ax1 = fig.add_subplot(gs[0, 1])
f_ax2 = fig.add_subplot(gs[1, :])

ax = np.array([f_ax0, f_ax1, f_ax2])
ax[0].imshow(trueModel)
ax[0].set(title='True')
ax[1].set(title='Inferred')
ax[1].imshow(inferredModel)

ax[2].plot(
    true_params, inferred_params, ls='none', marker='.',
    label=r'$R^{2}=$' + '{:.3f}'.format(r2_val))
ax[2].set(
    xlim=(true_params.min(), true_params.max()),
    ylim=(true_params.min(), true_params.max())
    )
ax[2].plot(
    [true_params.min(), true_params.max()],
    [true_params.min(), true_params.max()],
    # [0, 0],
    marker=',',
    color='k')
ax[2].legend()

plt.show()
