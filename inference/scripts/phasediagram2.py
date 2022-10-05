import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# from matplotlib import gridspec
# from scipy.interpolate import UnivariateSpline

import inference.analysis.new as analysis
# import inference.analysis.planalysis as planalysis
from inference import tools

# wow I've just found a way to pick? obs[Jrange, Trange]
parser = argparse.ArgumentParser()
parser.add_argument('--rundirs', nargs='*', required=True)

args = parser.parse_args()
# plt.style.use('~/Devel/styles/custom.mplstyle')
run_dirs = args.rundirs

observableNames = ['m', 'q', 'chiF', 'chiSG', 'e', 'tau']
observableLabels = [
    r'$|m|$', r'$q$',
    r'$\chi _{F}$', r'$\chi _{SG}$',
    r'$\epsilon$', r'$\tau$']
parameterNames = ['T', 'J']
plots = [
    # 'Full',
    # 'ISFs',
    'PD',
    # 'ObsCuts',
    # 'Corrs-avrgs',
    'Corrs',
    # 'Distributions',
    ]
pd = analysis.PhaseDiagram(run_dirs)
pd.calculate(obs_kwrds=['basic', 'tau'])

params, obs, ISFs = pd.load_run(runID=0)
# params, obs = pd.averages()
# obs['e'][obs['e'] >= 1] = 1
# obs['tau'][obs['tau'] <= 1] = 1

obs['e'][np.isnan(obs['e'])] = np.nanmax(obs['e'])

params = analysis.set_dimensions(params, 0, None, 0, None)
obs = analysis.set_dimensions(obs, 0, None, 0, None)

nStatepoints, B = ISFs.shape
print(nStatepoints, B)
square_length = int(np.sqrt(nStatepoints))
nJ_obs, nT_obs = params.shape
# N = 200
ISFs = ISFs.reshape((square_length, square_length, B))
# obs['chiF'] = obs['chiF'] * N
# params = params[0: 12, 0: None]
# obs = obs[0: 12, 0: None]
# I can't have a correaltion time less than 1 at my resolution!!
# so I should probably set something like...
# I think a surface might be more appropriate in the end...

# B_independent = B / obs['tau']
# B_independent[B_independent > B] = B

if 'ISFs' in plots:
    print(ISFs.shape)
    lags = np.arange(0, B)  # in units of per dump_freq
    fig, ax = plt.subplots()
    cols = plt.cm.get_cmap('viridis')(np.linspace(0, 1, nT_obs))
    marks = ['o', 'v', 's']
    Tmin = 0
    Tmax = 21
    Jmin = 10
    Jmax = 11
    J_range = np.arange(Jmin, Jmax)
    # J_range = [0, 20]
    T_range = np.arange(Tmin, Tmax)
    mi = 0
    for J_i in J_range:
        Ts = params['T'][J_i, 0:21]
        J = params['J'][J_i, 0]
        dump_freqs = params['cycles'][J_i, 0:21]
        # lags = np.arange(0, nSamples) * dump_freqs
        ISF = ISFs[J_i, 0:21, :]
        print(Ts)
        print(J)
        print(dump_freqs)
        print(ISF.shape)
        for T_i in T_range:
            lbl = "J={} | T={}".format(J, Ts[T_i])
            real_lags = lags * dump_freqs[T_i]
            ISF_at_T = ISF[T_i]
            ax.plot(
                real_lags, ISF_at_T / ISF_at_T.max(),
                marker=marks[mi],
                markeredgecolor='none',
                c=cols[T_i],
                label=lbl)
        mi += 1
    ax.set(
        xscale='log',
        xlabel=r'$\Delta t$ (MCCs)',
        ylabel=r'$C _{t}( \Delta t)$',
        # ylim=[0, 0.6]
        )
    # plt.legend()
    plt.show()

if 'PD' in plots:
    fig, axs = plt.subplots(3, 2, figsize=(16, 10))
    axs = axs.ravel()
    i = 0
    for ax, obsName, label in zip(axs, observableNames, observableLabels):
        data = obs[obsName]
        if i <= 1:
            norm = None
        else:
            norm = colors.LogNorm(data.min(), vmax=data.max())
        im = ax.pcolormesh(
            params['J'], params['T'], data,
            shading='auto',
            norm=norm
            )
        ax.set(title=label)
        fig.colorbar(im, ax=ax)
        # tools.plot_add_SK_analytical_boundaries(
        #     ax, np.min(params['J']), np.max(params['J']),
        #     np.min(params['T']), np.max(params['T']),
        # )
        i += 1
    plt.tight_layout()
    plt.show()

if 'ObsCuts' in plots:
    low_start = 0
    low_end = 8
    high_start = 14
    high_end = 20
    nTs = 21
    cols = plt.cm.get_cmap('viridis')(np.linspace(0, 1, nJ_obs))
    handles = []

    low_avrg = np.zeros((nTs), dtype=obs.dtype)

    # let's normalise them for easier comparison?
    # obs['chiF'] = obs['chiF'] / obs['chiF'].max()
    # obs['chiSG'] = obs['chiSG'] / obs['chiSG'].max()
    # obs['tau'] = obs['tau'] / obs['tau'].max()

    # binning for J/T datapoints
    J_over_T_max = (params['J'] / params['T']).max()
    J_over_T_min = (params['J'] / params['T']).min()
    bin_edges = np.linspace(J_over_T_min, J_over_T_max, 30)
    high_avrg = np.zeros((bin_edges.size - 1), dtype=obs.dtype)

    for obskwrd in observableNames:
        print(obskwrd)
        low_avrg[obskwrd] = np.mean(obs[obskwrd][0: low_end, :], axis=0)
        xdata = (
            params['J'][high_start: nTs, :] /
            params['T'][high_start: nTs, :])
        ydata = obs[obskwrd][high_start: nTs, :]
        bin_centres, high_avrg[obskwrd] = tools.bin_mean_xy(
            xdata, ydata, bin_edges)

    fig, ax = plt.subplots(3, 2, sharex='col', sharey='row')
    style_dict = {'alpha': 0.25, 'ls': 'none', "markeredgecolor": 'none'}
    for Ji in range(low_start, low_end + 1):
        xi = 1 / params['T'][Ji, :]
        print(params['J'][Ji, 0])
        ax[0, 0].plot(xi, obs['m'][Ji, :], c='tab:blue', **style_dict)
        ax[0, 0].plot(xi, obs['q'][Ji, :], c='tab:orange', **style_dict)
        ax[1, 0].plot(xi, obs['chiF'][Ji, :], c='tab:blue', **style_dict)
        ax[1, 0].plot(xi, obs['chiSG'][Ji, :], c='tab:orange', **style_dict)
        ax[1, 0].plot(xi, obs['tau'][Ji, :], c='tab:green', **style_dict)
        ax[2, 0].plot(xi, obs['e'][Ji, :], c='tab:blue', **style_dict)

    l_m, = ax[0, 0].plot(xi, low_avrg['m'], label=observableLabels[0])
    l_q, = ax[0, 0].plot(xi, low_avrg['q'], label=observableLabels[1])
    ax[0, 0].legend(handles=[l_m, l_q])

    l_chiF, = ax[1, 0].plot(xi, low_avrg['chiF'], label=observableLabels[2])
    l_chiSG, = ax[1, 0].plot(xi, low_avrg['chiSG'], label=observableLabels[3])
    l_tau, = ax[1, 0].plot(xi, low_avrg['tau'], label=observableLabels[5])
    ax[1, 0].legend(handles=[l_chiF, l_chiSG, l_tau])

    l_e, = ax[2, 0].plot(xi, low_avrg['e'], label=observableLabels[4])
    ax[2, 0].legend(handles=[l_e])

    col1_title = (
        r'$J=\{$' +
        '{} - {}'.format(params['J'][low_start, 0], params['J'][low_end, 0])
        + r'$\}$')
    ax[0, 0].set(title=col1_title)
    # analytical boundaries
    FFdash_Jbounds = np.array([1, 1.146])
    FFdash_Tbounds = np.array([1, 0.5077])
    J_over_Tbounds = FFdash_Jbounds / FFdash_Tbounds
    print(J_over_Tbounds)
    for ri in range(0, 3):
        ax[ri, 0].axvline(1, c='k', marker=',')
    for ri in range(0, 3):
        ax[ri, 1].axvline(1, c='k', marker=',')
        ax[ri, 1].axvspan(
            J_over_Tbounds[0], J_over_Tbounds[1], alpha=0.3, color='grey')

    # J, T for Tmin=0.5
    # 0.993031358885  0.999974932946
    # 1.14634146341  0.507720652746
    print('----')
    for Ji in range(high_start, high_end + 1):
        xi = params['J'][Ji, :] / params['T'][Ji, :]
        print(params['J'][Ji, 0])
        # xi = 1 / params['T'][Ji, :]
        # xi = params['T'][Ji, :]
        ax[0, 1].plot(xi, obs['m'][Ji, :], c='tab:blue', **style_dict)
        ax[0, 1].plot(xi, obs['q'][Ji, :], c='tab:orange', **style_dict)
        ax[1, 1].plot(xi, obs['chiF'][Ji, :], c='tab:blue', **style_dict)
        ax[1, 1].plot(xi, obs['chiSG'][Ji, :], c='tab:orange', **style_dict)
        ax[1, 1].plot(xi, obs['tau'][Ji, :], c='tab:green', **style_dict)
        ax[2, 1].plot(xi, obs['e'][Ji, :], c='tab:blue', **style_dict)

    xi = bin_centres
    l_m, = ax[0, 1].plot(xi, high_avrg['m'], label=observableLabels[0])
    l_q, = ax[0, 1].plot(xi, high_avrg['q'], label=observableLabels[1])
    ax[0, 1].legend(handles=[l_m, l_q])

    l_chiF, = ax[1, 1].plot(xi, high_avrg['chiF'], label=observableLabels[2])
    l_chiSG, = ax[1, 1].plot(xi, high_avrg['chiSG'], label=observableLabels[3])
    l_tau, = ax[1, 1].plot(xi, high_avrg['tau'], label=observableLabels[5])
    ax[1, 1].legend(handles=[l_chiF, l_chiSG, l_tau])

    l_e, = ax[2, 1].plot(xi, high_avrg['e'], label=observableLabels[4])
    ax[2, 1].legend(handles=[l_e])

    col1_title = (
        r'$J=\{$' +
        '{} - {}'.format(params['J'][high_start, 0], params['J'][high_end, 0])
        + r'$\}$')
    ax[0, 1].set(title=col1_title)

    ax[2, 0].set(xlabel=r'$1 / T$')
    ax[2, 1].set(xlabel=r'$J / T$')
    ax[0, 0].set(ylabel=r'$\mathcal{O}$')
    ax[1, 0].set(ylabel=r'$d \mathcal{O} / d \mathcal{F}$')
    ax[2, 0].set(ylabel=r'$\epsilon$')
    for a in ax.ravel():
        a.set(yscale='log')

    # fig.legend(handles=handles, loc='upper right')
    # plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    plt.show()

if 'Corrs-avrgs' in plots:
    i = 0
    # low
    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].plot(low_avrg['chiSG'], low_avrg['e'], ls='none', c=cols[i])
    ax[1].plot(low_avrg['tau'], low_avrg['e'], ls='none', c=cols[i])
    ax[2].plot(low_avrg['chiSG'], low_avrg['e'], ls='none', c=cols[i])
    ax[3].plot(low_avrg['tau'], low_avrg['e'], ls='none', c=cols[i])

    # i = -1
    # ax[0].plot(high_avrg['chiSG'], high_avrg['e'], ls='none', c=cols[i])
    # ax[1].plot(high_avrg['tau'], high_avrg['e'], ls='none', c=cols[i])
    # ax[2].plot(high_avrg['chiSG'], high_avrg['e'], ls='none', c=cols[i])
    # ax[3].plot(high_avrg['tau'], high_avrg['e'], ls='none', c=cols[i])

    ax[0].set(xlabel=observableLabels[3], ylabel=observableLabels[4])
    ax[0].set(xscale='log', yscale='linear')
    ax[0].set_ylim(bottom=0.25, top=0.6)
    ax[1].set(xlabel=observableLabels[5], ylabel=observableLabels[4])
    ax[1].set(xscale='log', yscale='log')
    ax[2].set(xlabel=observableLabels[3], ylabel=observableLabels[4])
    ax[2].set(xscale='log', yscale='log')
    ax[3].set(xlabel=observableLabels[5], ylabel=observableLabels[4])
    ax[3].set(xscale='log', yscale='linear')
    ax[3].set_ylim(bottom=0.25, top=0.6)
    plt.tight_layout()
    plt.show()

if 'Corrs' in plots:
    # fig, ax = plt.subplots(2, 2)
    fig, ax = plt.subplots()
    ax = [ax]
    # ax = ax.ravel()
    nJ_obs = 5
    cols = plt.cm.get_cmap('viridis')(np.linspace(0, 1, nJ_obs))
    for i in range(0, nJ_obs):
        B_independent = B / obs['tau'][i, :]
        B_independent[B_independent > B] = B
        ax[0].plot(
                1 / params['T'][i, :], obs['e'][i, :],
                # obs['chiSG'][i, :], obs['e'][i, :],
                # ls='none',
                c=cols[i])
        ax[0].set(xlabel='1/T', ylabel=observableLabels[4])
        # ax[0].set(xlabel=observableLabels[3], ylabel=observableLabels[4])
        ax[0].set(xscale='linear', yscale='linear')
        # ax[0].set_ylim(bottom=0.25, top=0.6)

        # ax[1].plot(
        #         obs['tau'][i, :], obs['e'][i, :],
        #         ls='none', c=cols[i])
        # ax[1].set(xlabel=observableLabels[5], ylabel=observableLabels[4])
        # ax[1].set(xscale='log', yscale='log')
        # # ax[1].set_ylim(bottom=0.25, top=0.6)
        # # ax[1].set_xlim(left=0.2, right=50)

        # ax[2].plot(
        #         obs['chiSG'][i, :], obs['e'][i, :],
        #         ls='none', c=cols[i])
        # ax[2].set(xlabel=observableLabels[3], ylabel=observableLabels[4])
        # ax[2].set(xscale='log', yscale='log')

        # # ax[2].set_ylim(bottom=0.25, top=0.6)
        # # ax[2].set_xlim(left=0.1, right=1)
        # # let's try make this a bit clearer
        # # gotta work on the splitting graph!
        # # B_independent
        # ax[3].plot(
        #         obs['tau'][i, :], obs['e'][i, :],
        #         ls='none', c=cols[i])
        # ax[3].set(xlabel=observableLabels[5], ylabel=observableLabels[4])
        # ax[3].set(xscale='log', yscale='log')
    plt.show()

if 'Distributions' in plots:
    nbins = 50
    fig, axs = plt.subplots(3, 2)  # figsize=(16, 10)
    axs = axs.ravel()
    for ax, obsName, label in zip(axs, observableNames, observableLabels):
        flat_obs = obs[obsName].ravel()
        ax.hist(
            flat_obs, bins=nbins,
            weights=np.ones_like(flat_obs) / len(flat_obs))
        ax.set(title=label)
        ax.set(yscale='log')
    plt.show()
