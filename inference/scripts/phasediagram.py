import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import gridspec

import inference.analysis.new as analysis
# import inference.analysis.planalysis as planalysis

from inference import tools


parser = argparse.ArgumentParser()
parser.add_argument('--rundirs', nargs='*', required=True)

args = parser.parse_args()
plt.style.use('~/Devel/styles/custom.mplstyle')
run_dirs = args.rundirs

pd = analysis.PhaseDiagram(run_dirs, recalculate=False)
pd.load()
parameters = pd.parameters
obs = pd.observables

obs_avrg, obs_std = pd.calc_averages()

obs_labels = [
    r'$|m|$', r'$q$', r'$\chi _{F}$', r'$\chi _{SG}$', r'$\epsilon$', r'$Q$']
plots = [
    # 'Averages',
    # 'Boundaries',
    # 'Smoothed',
    # 'IsoLines',
    # 'MinTracker',  # still need to think about this plot!
    # 'RepOverview'
    # '3D',
    # 'Contour',
    'Correlation',
    'Collapsed',
    # 'Simple',
    ]

if 'Averages' in plots:
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=3,)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1]))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[1, 1]))
    ax.append(fig.add_subplot(gs[0, 2]))
    ax.append(fig.add_subplot(gs[1, 2]))
    # ax.append(fig.add_subplot(gs[0:, 2:]))

    OP_max = np.max([obs_avrg[0].max(), obs_avrg[1].max()])
    OP_min = np.max([obs_avrg[0].min(), obs_avrg[1].min()])
    print(OP_min, OP_max)
    SUSCEPT_max = np.max([obs_avrg[2].max(), obs_avrg[3].max()])
    SUSCEPT_min = np.max([obs_avrg[2].min(), obs_avrg[3].min()])
    print(SUSCEPT_min, SUSCEPT_max)

    for k in range(0, 6):
        data = obs_avrg[k]
        # data[data <= 0] = 0
        if k < 2:
            im = ax[k].pcolormesh(
                parameters[2], parameters[0], data,
                # vmin=OP_min, vmax=OP_max,
                norm=colors.LogNorm(vmin=OP_min, vmax=OP_max),
                shading='auto')
        if 2 <= k < 4:
            im = ax[k].pcolormesh(
                parameters[2], parameters[0], data,
                # vmin=SUSCEPT_min, vmax=SUSCEPT_max,
                norm=colors.LogNorm(vmin=SUSCEPT_min, vmax=SUSCEPT_max),
                shading='auto')
        if k == 4:
            im = ax[k].pcolormesh(
                parameters[2], parameters[0], data,
                norm=colors.LogNorm(vmin=data.min(), vmax=data.max()),
                shading='auto')
        if k > 4:
            im = ax[k].pcolormesh(
                parameters[2], parameters[0], data, shading='auto')
        plt.colorbar(im, ax=ax[k])

        tools.plot_add_SK_analytical_boundaries(
            ax[k],
            np.min(parameters[2]), np.max(parameters[2]),
            np.min(parameters[0]), np.max(parameters[0]),
            )
        ax[k].set(title="{}".format(obs_labels[k]))
        ax[k].set(xlabel=r'$\mu \left( J \right)$', ylabel=r'$T$')

    plt.tight_layout()
    plt.show()

if 'Boundaries' in plots:
    print("!!!")
    Js = parameters[2][0, :]  # x
    Ts = parameters[0][:, 0]  # y
    minError_i = np.argwhere(obs_avrg[4] == obs_avrg[4].min())
    # I NEED TO REMEMBER HOW TO DO THIS!
    T_minError = parameters[0][obs_avrg[4] == obs_avrg[4].min()]
    J_minError = parameters[3][obs_avrg[4] == obs_avrg[4].min()]
    print(T_minError, J_minError)
    vmax = 1
    obs_avrg[4][obs_avrg[4] >= vmax] = vmax
    fig, ax = plt.subplots()
    im = ax.pcolormesh(
        parameters[2], parameters[0], obs_avrg[4], shading='auto',
        norm=colors.LogNorm(vmin=obs_avrg[4].min(), vmax=obs_avrg[4].max())
        )
    plt.plot(
        J_minError, T_minError,
        marker='*', markersize='30', markerfacecolor='r',
        linestyle='none',
        label=r'$min _{\epsilon}$')
    tools.plot_add_SK_analytical_boundaries(
            ax,
            np.min(parameters[2]), np.max(parameters[2]),
            np.min(parameters[0]), np.max(parameters[0]),
            )

    chiPicks = [2, 3]
    labels = [r'$\chi _{F}$', r'$\chi _{SG}$']
    for chiPick in chiPicks:
        Tmaxs = []
        for isoPick in range(0, len(Js)):
            obs_IsoJs = obs_avrg[:, :, isoPick]
            chi = obs_IsoJs[chiPick]
            i_maxChi = np.argmax(chi)
            T_at_max = Ts[i_maxChi]
            Tmaxs.append(T_at_max)
        ax.plot(Js, Tmaxs, label=labels[chiPick-2])
    ax.legend()
    ax.set(label=r'$\mu \left( J \right)$', ylabel=r'$T$')
    ax.set(title=r'Error $\epsilon$, $N=200$')
    fig.colorbar(im, ax=ax)
    plt.show()

if 'Smoothed' in plots:
    xx = parameters[2]
    yy = parameters[0]
    z = obs_avrg[4]
    xx_smooth, yy_smooth, z_smooth = tools.interpolateRBF(
        xx, yy, z, smoothing=0)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.ravel()
    im = ax[0].pcolormesh(
        xx, yy, z, shading='auto',
        norm=colors.LogNorm(vmin=obs_avrg[4].min(), vmax=obs_avrg[4].max())
        )
    fig.colorbar(im, ax=ax[0])
    im = ax[1].pcolormesh(
        xx_smooth, yy_smooth, z_smooth, shading='auto',
        norm=colors.LogNorm(vmin=obs_avrg[4].min(), vmax=obs_avrg[4].max())
        )
    fig.colorbar(im, ax=ax[1])
    plt.show()

if 'RepOverview' in plots:
    selecta = [0, 1, 2, 3, 4]
    fig, ax = plt.subplots(len(selecta), 5, figsize=(16, 8))
    ax = ax.ravel()
    titles = [r"$run_{0}$", r"$run_{1}$", r"$run_{2}$", r"$\mu$", r"$\sigma$"]

    ax_index = 0
    for i in selecta:
        for r in range(0, 3):
            data = obs[i][r]
            ax[ax_index].imshow(data, origin='lower')
            ax[ax_index].set(title=titles[r])
            if r == 0:
                ax[ax_index].text(
                    0.1, 0.8, obs_labels[i],
                    bbox={'facecolor': 'white', 'pad': 5},
                    transform=ax[ax_index].transAxes)
            ax_index += 1
        data = obs_avrg[i]
        ax[ax_index].imshow(data, origin='lower')
        ax[ax_index].set(title="{}".format(titles[3]))
        ax_index += 1
        data = obs_std[i]
        ax[ax_index].imshow(data, origin='lower')
        ax[ax_index].set(title="{}".format(titles[4]))
        ax_index += 1
    plt.show()


if 'IsoLines' in plots:
    vmax = 0.5
    obs_avrg[4][obs_avrg[4] >= vmax] = vmax
    # cheeky smoothing in here?
    # xx = parameters[2]
    # yy = parameters[0]
    # z = obs_avrg[4]
    # xx_smooth, yy_smooth, z_smooth = tools.interpolateRBF(
    #    xx, yy, z, smoothing=0)

    Js = parameters[2][0, :]  # x
    Ts = parameters[0][:, 0]  # y
    print(Js)
    print(Ts)
    fig, ax = plt.subplots(2, 1)
    ax = ax.ravel()
    isoPicks = np.arange(0, 5)
    for isoPick in isoPicks:
        obs_IsoJs = obs_avrg[:, :, isoPick]
        obs_IsoTs = obs_avrg[:, isoPick, :]

        # = np.max(obs_IsoJs[2])
        chi_choice = 4
        i_maxChi = np.argmax(obs_IsoJs[chi_choice])
        print(i_maxChi)

        ax[0].plot(
            Ts,
            obs_IsoJs[chi_choice],
            label=obs_labels[chi_choice])

        i = 3
        ax[1].plot(
            Ts,
            obs_IsoJs[i],
            label=obs_labels[i])
        ax[1].axvline(Ts[i_maxChi], marker=',')
        # ax[1].plot(Js, T0_const)

    ax[0].legend()
    ax[1].legend()
    plt.show()

if '3D' in plots:
    vmax = 0.5
    obs_avrg[4][obs_avrg[4] >= vmax] = vmax
    obs_avrg[3][obs_avrg[3] >= vmax] = vmax
    xx = parameters[2]
    yy = parameters[0]
    z = obs_avrg[4]
    xx_smooth, yy_smooth, z_smooth = tools.interpolateRBF(
        xx, yy, z, smoothing=0.0)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax2 = fig.add_axes([0.4, 0.6, 0.45, 0.3])
    # vmax = np.max(obs_avrg[4]),
    # obs_avrg[3][obs_avrg[3] > vmax] = vmax

    ax.plot_surface(
        xx, yy, z,
        color='tab:blue',
        alpha=0.5,
        linewidth=0, antialiased=False)
    # ax.plot_surface(
    #     xx_smooth, yy_smooth, z_smooth,
    #     color='tab:orange',
    #     alpha=0.5,
    #     linewidth=0, antialiased=False)
    ax.plot_surface(
        xx, yy, obs_avrg[3],
        color='tab:orange',
        alpha=0.5,
        linewidth=0, antialiased=False)

    # ax.set_zlim(np.min(obs_avrg[4]), np.max(obs_avrg[4]))
    # ax.set_zlim(0.25, 1)
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.set(xlabel='J', ylabel='T', zlabel='Obs')

    ax.text2D(
        0.05, 0.9, obs_labels[3], c='tab:blue', transform=ax.transAxes)
    ax.text2D(
        0.15, 0.9, obs_labels[4], c='tab:orange', transform=ax.transAxes)
    plt.show()

if 'Contour' in plots:
    vmin = 0
    vmax = 0.1

    i = 5
    data = obs_avrg[i]
    data[data <= vmin] = vmin
    # data[data >= vmax] = vmax

    # log scales?

    xx = parameters[2]
    yy = parameters[0]
    z = data
    s = 0.001
    xx_smooth, yy_smooth, z_smooth = tools.interpolateRBF(
        xx, yy, z, smoothing=s)

    norm = colors.LogNorm(vmin=data.min(), vmax=data.max())
    cm = plt.cm.gray.reversed()
    fig, ax = plt.subplots(2, 1)
    ax = ax.ravel()

    im = ax[0].pcolormesh(
        xx, yy, z,
        shading='auto',
        # norm=norm,
        cmap=cm)
    cs = ax[0].contour(
        xx, yy, z,
        # norm=norm
        )
    ax[0].clabel(
        cs, inline=True, fontsize=10)
    plt.colorbar(im, ax=ax[0])

    im = ax[1].pcolormesh(
        xx_smooth, yy_smooth, z_smooth,
        shading='auto',
        # norm=norm,
        cmap=cm)
    cs = ax[1].contour(
        xx_smooth, yy_smooth, z_smooth,
        # norm=norm
        )
    ax[1].clabel(cs, inline=True, fontsize=10)
    plt.colorbar(im, ax=ax[1])
    tools.plot_add_SK_analytical_boundaries(
            ax[1],
            np.min(xx), np.max(xx),
            np.min(yy), np.max(yy),
            )

    ax[0].set(xlabel='J', ylabel='T', title='RAW R^2')
    ax[1].set(xlabel='J', ylabel='T', title='BLURRED s={} R^2'.format(s))
    # ax[0].axis('square')
    # ax[1].axis('square')
    # plt.tight_layout()
    plt.show()

if 'Simple' in plots:
    i = 5
    data = obs_avrg[i]
    vmin = np.min(data[:, 0:1])
    vmin = 0
    data[data <= vmin] = vmin

    fig, ax = plt.subplots()
    for i in range(0, 21):
        ax.plot(data[:, i])
    plt.show

    fig, ax = plt.subplots()
    im = ax.imshow(data, origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='J', ylabel='T')
    plt.show()

x_cut = None
obs_avrg = obs_avrg[:, x_cut:, :]
parameters = parameters[:, x_cut:, :]
if 'Correlation' in plots:
    print('Correlation Plot')
    vmin = 0
    vmax = 0.5
    # obs_avrg[obs_avrg <= vmin] = vmin
    # obs_avrg[obs_avrg >= vmax] = vmax
    # obs_avrg = obs_avrg[:, :, 0:10]
    # I want to do the same for the collapsed stuff?
    obs_x = 3
    obs_y = 4
    x = obs_avrg[obs_x]
    y = obs_avrg[obs_y]

    fig, ax = plt.subplots()
    J_picks = np.arange(0, 5)
    n = J_picks.size
    # I want all the assoicated xs?
    print(x[:, J_picks].shape, y[:, J_picks].shape)
    x_data = x[:, J_picks].ravel()
    y_data = y[:, J_picks].ravel()
    # hmm not sure this fitting stuff really helped me...
    # maybe I should just have stuck to calculating the avergaes lol
    # x_fit, y_fit = tools.KKR(x_data, y_data)
    # plt.plot(x_fit, y_fit, c='k', marker=',')
    cols = plt.cm.get_cmap('viridis')(np.linspace(0, 1, n))
    for ci, i in enumerate(J_picks):
        label = 'J = {}'.format(parameters[2][0, i])
        # print(parameters[0][5:, i])
        xi = x[:, i]
        yi = y[:, i]
        analysis.plotCorrelation(
            ax, xi, yi,
            obs_labels[obs_x], obs_labels[obs_y],
            label=label,
            ls='none',
            color=cols[ci])
    
    # ax.set(ylim=[y.min(), y.max()])
    ax.set_xscale('log')
    ax.set_yscale('log')

    # print(y[:, J_picks].shape)
    # thing = np.swapaxes(y[:, J_picks], 0, 1)
    # print(thing.shape)
    # bins = np.linspace(x[:, ].min(), xis.max(), 20)
    # tools.plot_binned_errorbars(ax, xis, yis, bins)
    # plt.ylim(y[:, 0:n].min(), 1)
    # plt.xlim(0, 0.1)
    plt.legend(bbox_to_anchor=(1.05, 1.))
    plt.show()

if 'Collapsed' in plots:
    print('Collapsed Plot')
    obs_pick = 3
    fig, axs = plt.subplots(3, 2)
    axs = axs.ravel()
    for obs_pick in range(0, 6):
        ax = axs[obs_pick]
        data = obs_avrg[obs_pick, :, :]
        temps = parameters[0, :, :]
        Js = parameters[2, :, :]
        J_over_T = Js / temps
        # fig, ax = plt.subplots()
        J_picks = np.arange(15, 21)
        n = J_picks.size
        cols = plt.cm.get_cmap('viridis')(np.linspace(0, 1, n))
        xis = []
        yis = []
        lines = []
        labels = []
        for ci, i in enumerate(J_picks):
            xi = J_over_T[:, i]
            # xi = temps[:, i]
            yi = data[:, i]
            label = 'J = {}'.format(Js[0, i])
            labels.append(label)
            line = ax.plot(
                xi, yi,
                ls='none',
                c=cols[ci], label=label)[0]
            lines.append(line)
            xis.append(xi)
            yis.append(yi)
        xis = np.array(xis)
        yis = np.array(yis)

        xis = xis.ravel()
        yis = yis.ravel()
        bins = np.linspace(xis.min(), xis.max(), 20)
        tools.plot_binned_errorbars(ax, xis, yis, bins)

        ax.set(xlabel='J/T')
        ax.set(ylabel=obs_labels[obs_pick])
    # plt.legend()
    fig.legend(
        lines,
        labels,
        loc="center right",
        # bbox_to_anchor=(1.05, 0.8)
        )
    plt.tight_layout()
    plt.show()
# analysis.plotCorrelation(
#     ax, xi, yi,
#     obs_labels[obs_x], obs_labels[obs_y],
#     # label=label,
#     ls='none',
#     color=cols[ci])
# label = 'J = {}'.format(parameters[2][0, i])
# xi = x[:, i]
# yi = y[:, i]
# ax.set_yscale('log')
    # so how do I cut stuff now..?

    # # hmm I'm a little bit confused not gonna lie...
    # nObs, nXpoints, nYpoints = data.shape
    # print(nObs, nXpoints, nYpoints)
    # flat_data = np.reshape(data, (nObs, nXpoints * nYpoints))

    # J_over_T = params[2, :, :] / params[0, :, :]
    # J_over_T_flat = J_over_T.ravel()

    # print(data.shape, flat_data.shape)
    # print(J_over_T.shape, J_over_T_flat.shape)
    # obs_pick = 5
    # plt.plot(J_over_T_flat, flat_data[obs_pick], ls='none')
    # plt.xlabel(r'$\frac{J}{T}$')
    # plt.ylabel(obs_labels[obs_pick])
    # plt.show()

    # I want to do a plots of J/T vs my observables.
    # this should be easy right?

# tools.plot_beautiful_errorbars(ax[3], xs, datasets2)

# error susceptiblity corrleaiton plot log-log check for power
# how many configs per sample have the same q
# something about how to measure similarity of states
# I should correlate Q and eps!