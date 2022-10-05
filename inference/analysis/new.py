import numpy as np
import glob
import os
import h5py
import lzma
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from numpy.core.numeric import correlate
from os.path import join
from numpy.core.numeric import outer
from scipy import signal
# from scipy.signal import signal.correlate

from scipy.interpolate import UnivariateSpline
from scipy.stats import pearsonr

from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
from tqdm import tqdm

import inference.io as iohdf5
import inference.analysis.planalysis as planalysis


h5py.get_config().track_order = True


def hdf5_plotObsAndFluc(data_fname):
    # Tc = 2.269J/kb!
    # think I want to call it recalc!
    lM = r'<$ |M| / N $>'
    lChi = r'$\chi ^{*} / N$'
    lE = r'<$E / N$>'
    lCv = r'$C_ {v} / N$'

    labels = [lM, lChi, lE, lCv]
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    for a, l in zip(ax.ravel(), labels):
        a.set(xlabel='T', ylabel=l)
        a.axvline(2.269, marker=',', c='k')
    with iohdf5.Readhdf5_mc(data_fname, prod_only=True) as fin:
        metadata = fin.get_metadata()
        configs_all_T = fin.read_many_datasets("configurations")
        energy_all_T = fin.read_many_datasets("energy")

    sweep_paramters = np.array(metadata["SweepParameterValues"])
    # print(sweep_paramters.shape)
    # I THINK I SAVE E AS E/T * NEED TO TIMES IT BY T TO MAKE SURE
    # THAT NOTHING BAD IS HAPPENING!
    # YP SAVING E/T ATM KEEP THIS IN MIND!!!!
    output_aray = np.empty((4, sweep_paramters.size))

    for c, sweep_param in enumerate(sweep_paramters):
        print(c, sweep_param)
        # Ts = np.array(md["SweepParameterValues"])
        N = metadata["SystemSize"]
        # print(configs_all_T[c].shape)
        M_traj = np.sum(configs_all_T[c], axis=1)
        Mabs_traj = abs(M_traj)
        Mabs_mean = np.mean(Mabs_traj) / N
        Mabs_fluct = np.var(Mabs_traj) / sweep_param

        E_traj = energy_all_T[c] * sweep_param
        E_mean = np.mean(E_traj) / N
        E_fluct = np.var(E_traj) / (sweep_param ** 2)
        output_aray[0, c] = Mabs_mean
        output_aray[1, c] = Mabs_fluct
        output_aray[2, c] = E_mean
        output_aray[3, c] = E_fluct

    ax[0, 0].plot(
        sweep_paramters, output_aray[0, :], '.', label='N = {}'.format(N))
    ax[0, 1].plot(sweep_paramters, output_aray[1, :] / N, '.')
    ax[1, 0].plot(sweep_paramters, output_aray[2, :], '.')
    ax[1, 1].plot(sweep_paramters, output_aray[3, :] / N, '.')

    ax[0, 0].legend()
    plt.tight_layout()
    plt.show()


def hdf5_absmqchi(data_fname):
    # Tc = 2.269J/kb!
    # think I want to call it recalc!
    # lM = r'<$ |M| / N $>'
    # lChi = r'$\chi ^{*} / N$'
    # lE = r'<$E / N$>'
    # lCv = r'$C_ {v} / N$'

    # labels = [lM, lChi, lE, lCv]
    '''
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    for a, l in zip(ax.ravel(), labels):
        a.set(xlabel='T', ylabel=l)
        a.axvline(2.269, marker=',', c='k')
    '''
    with iohdf5.Readhdf5_mc(
            data_fname, show_metadata=False, prod_only=True) as fin:
        metadata = fin.get_metadata()
        configs_all_T = fin.read_many_datasets("configurations")
        energy_all_T = fin.read_many_datasets("energy")

    sweep_paramters = np.array(metadata["SweepParameterValues"])
    output_aray = np.empty((4, sweep_paramters.size))
    qs = []
    # fig, ax = plt.subplots(1, 2)
    # ax = ax.ravel()
    for c, sweep_param in enumerate(sweep_paramters):
        # print(c, sweep_param)
        # Ts = np.array(md["SweepParameterValues"])
        N = metadata["SystemSize"]
        # print(configs_all_T[c].shape)
        # I don't know what I was doing with the configs here
        # but this is a fucky fuck
        M_traj = np.sum(configs_all_T[c], axis=1)
        Mabs_traj = abs(M_traj)
        Mabs_mean = np.mean(Mabs_traj) / N
        Mabs_fluct = np.var(Mabs_traj) / sweep_param

        E_traj = energy_all_T[c] * sweep_param
        E_mean = np.mean(E_traj) / N
        E_fluct = np.var(E_traj) / (sweep_param ** 2)
        output_aray[0, c] = Mabs_mean
        output_aray[1, c] = Mabs_fluct
        output_aray[2, c] = E_mean
        output_aray[3, c] = E_fluct
        # I square too soon!
        # I should sum and then divide by N!!!
        # M = np.sum()
        # let's clean this stuff up!
        # print(c)
        Si_avrg = np.mean(configs_all_T[c], axis=0)
        Si_avrg_sqr = Si_avrg ** 2
        m = np.mean(Si_avrg)
        q = np.mean(Si_avrg_sqr)
        # print(Si_avrg[0:4])
        # print(Si_avrg_sqr[0:4])
        # ax[0].plot(m, ls='none')
        # ax[1].plot(q, ls='none')
        output_aray[0, c] = np.mean(Si_avrg)
        # print('---')
        # print(Si_avrg.shape)
        # print(Si_avrg[0:10])
        # print(np.mean(Si_avrg))
        # print(configs_all_T[c].shape)
        # print(configs_all_T[c][:10, 0])
        x = np.sum(configs_all_T[c][:10, :], axis=0)
        # print(x)
        x = x ** 2
        # print(x)
        # si_averages = np.mean(configs_all_T[c], axis=0)
        # q = np.mean(x) / N
        qs.append(q)
    # plt.show()
    '''
    ax[0, 0].plot(
        sweep_paramters, output_aray[0, :], '.', label='N = {}'.format(N))
    ax[0, 1].plot(sweep_paramters, output_aray[1, :] / N, '.')
    ax[1, 0].plot(sweep_paramters, output_aray[2, :], '.')
    ax[1, 1].plot(sweep_paramters, output_aray[3, :] / N, '.')
    ax[0, 0].legend()
    '''
    return output_aray[0, :], np.array(qs), output_aray[1, :] / N


# q = 1/N * sum_N ((<s_i>)^2)
def mq(config_trajectory):
    # print('Hi')
    # print(config_trajectory.shape)
    EXP_si = np.mean(config_trajectory, axis=0)
    # EXP_si_qrt = EXP_si ** 2
    m = np.mean(EXP_si)
    q = np.mean(EXP_si ** 2)
    return m, q
    # axis = 0 has configs, axis 1 has spins
    # for config in config_trajectory:
    #     print(config.shape)


# x and y 1D arrays? who knows maybe its working?
# C = <x * y> - <x><y>
def connected_correlation(x, y):
    A = np.mean(x * y)
    B = np.mean(x) * np.mean(y)
    return A - B


@njit
def mag(config_1, config_2):
    m1 = np.mean(config_1)
    m2 = np.mean(config_2)
    return m2 * m1


@njit
def m_autocorr(config_trajectory):
    B, N = config_trajectory.shape
    print(B, N)
    q_trajectory = np.zeros(B)
    normalisations = np.arange(B, 0, -1)
    B_delays = B
    for start_t in range(0, B):

        for delay in range(0, B_delays):

            q_trajectory[delay] += mag(
                config_trajectory[start_t, :],
                config_trajectory[start_t + delay, :])
        B_delays = int(B_delays - 1)
    q_trajectory = q_trajectory / normalisations

    return q_trajectory


@njit
def fast_tcorr(observable_trajectory):
    B = observable_trajectory.size

    # obs_product = 0
    # obs_bound_avrg = 0
    # obs_bound_avrg_delayed = 0
    autocorrelation = np.zeros(B)

    del_ts = np.arange(0, B)
    for del_t in del_ts:
        obs_product = 0
        obs_bound_avrg = 0
        obs_bound_avrg_delayed = 0
        sum_limit = B - del_t

        # integral:
        for start_t in range(0, sum_limit):
            obs = observable_trajectory[start_t]
            obs_delayed = observable_trajectory[start_t + del_t]

            obs_product += obs * obs_delayed
            obs_bound_avrg += obs
            obs_bound_avrg_delayed += obs_delayed

        obs_mean = obs_product / sum_limit
        obs_bound_avrg = obs_bound_avrg / sum_limit
        obs_bound_avrg_delayed = obs_bound_avrg_delayed / sum_limit

        autocorrelation[del_t] = (
            obs_mean - (obs_bound_avrg * obs_bound_avrg_delayed))

    autocorrelation_norm = np.max(autocorrelation)
    autocorrelation = autocorrelation / autocorrelation_norm
    return del_ts, autocorrelation


# I should read fisher and hertz!!! and cite this for my observables
# definitions!
class Observables():
    def __init__(self, config_trajectory, T=1):
        B, N = config_trajectory.shape
        self.nsamples = B
        self.nspins = N
        self.temp = T
        self.configs = config_trajectory

        si_avrg = np.mean(config_trajectory, axis=0)
        si_avrg_sqr = si_avrg ** 2

        self.m = np.mean(si_avrg)
        self.q = np.mean(si_avrg_sqr)

        m_trajectory = np.mean(config_trajectory, axis=1)
        absm_trajectory = abs(m_trajectory)

        self.m_traj = m_trajectory
        self.absm_traj = absm_trajectory
        self.absm = np.mean(absm_trajectory)

        chi = np.var(m_trajectory)
        chi_absm = np.var(absm_trajectory)

        self.chis = chi, chi_absm

    def compute_time_correlation(self, calc_correlation_time=False):
        delays, autocorrelation = fast_tcorr(self.m_traj)
        if calc_correlation_time is True:
            ac_spline = UnivariateSpline(
                delays, autocorrelation-np.exp(-1), s=0)
            ac_roots = ac_spline.roots()
            correlation_time = ac_roots[0]
            return delays, autocorrelation, correlation_time
        return delays, autocorrelation

    def compute_spin_spin_correlation(self):
        cij = np.cov(self.configs.T)
        self.cij = cij
        chi = np.sum(cij) / (self.nspins * self.temp)
        cij_sg = np.sum(cij ** 2) / (self.nspins * (self.temp ** 2))

        return chi, cij_sg

    def scipy_autocorr(self, tol=-0.1, calc_correlation_time=False):

        x = self.m_traj
        xmean = x.mean()
        # not sure if I should use var or not!
        # xvar = x.var()
        # I could do x = config dot config something
        correlations = signal.correlate(x - xmean, x - xmean, mode="full")
        lags = signal.correlation_lags(x.size, x.size, mode="full")
        correlations = correlations[correlations.size//2:]
        correlations /= np.max(correlations)
        lags = lags[lags.size//2:]

        if calc_correlation_time is True:
            ac_spline = UnivariateSpline(
                lags, correlations-np.exp(-1), s=0)
            ac_roots = ac_spline.roots()
            correlation_time = ac_roots[0]
            return lags, correlations, correlation_time
        return lags, correlations



# I don't think what I calculate is the overlap actually :/
# not sure!!
# not sure how to optimize this to make it run in a reasonable time?
@njit
def overlap_trajectory_fast(config_trajectory):
    B, N = config_trajectory.shape
    # print(B, N)
    # B = 100
    q_trajectory = np.zeros(B)
    normalisations = np.arange(B, 0, -1)
    B_delays = B
    for start_t in range(0, B):
        # config_1 = config_trajectory[start_t, :]
        for delay in range(0, B_delays):
            # config_1 = config_trajectory[start_t, :]
            # config_2 = config_trajectory[start_t + delay, :]
            # print(config_1.sh)
            # q = np.dot(config_1, config_2) / N
            # q_trajectory[delay] = q
            # config_2 = config_trajectory[start_t + delay, :]
            # q = np.dot(config_1, config_2) / N
            # q_trajectory[delay] = q
            q_trajectory[delay] += overlap(
                config_trajectory[start_t, :],
                config_trajectory[start_t + delay, :])
        B_delays = int(B_delays - 1)
    q_trajectory = q_trajectory / normalisations

    return q_trajectory

# q is between two configurations!
@njit
def overlap(config_1, config_2):
    N = config_1.size
    q = np.dot(config_1, config_2)
    q /= N
    return q


def overlap_trajectory(config_trajectory):
    B, N = config_trajectory.shape
    print(B, N)
    # B = 1000
    q_trajectory = np.zeros(B)
    # delays = np.arange(0, B, dtype=int)
    # delays = delays[0:50]
    # I think t should be outside loop acutally!
    # for every t and every delay do a thing!
    # t_min = 0
    # iterator = (s.upper() for s in oldlist)
    normalisations = np.arange(B, 0, -1)
    # print(normalisations)
    B_delays = B
    for start_t in range(0, B):
        for delay in range(0, B_delays):
            q_trajectory[delay] += overlap(
                config_trajectory[start_t, :],
                config_trajectory[start_t + delay, :])
        B_delays = int(B_delays - 1)
    q_trajectory = q_trajectory / normalisations
        # do stuff
        # print(start_t)
        # t_min = int(t_min + 1)
    '''
    for delay in delays:
        for start_t in range(0, B):
            q_trajectory[delay] += overlap(
                config_trajectory[start_t, :],
                config_trajectory[start_t + delay, :])
        q_trajectory[delay] /= B
        B = int(B - 1)
    '''
    # still not 100% sure this is right way to do it!
    # this isn't quite right still!
    # reduce delays by 1
    # q should be a function of delays
    return q_trajectory


    # this is the dot product between the two?
    # no the mean of the stuff I think?
# as a function of t
# not sure about all this crap...
# I SHOULD CHANGE THIS TO JUST BE A FUNCTION OF LAG!
# THAT WILL MAKE THINGS EASIER AND CLEARER!
# no even better, should be a function of two configurations
def overlap_bad(config_trajectory):
    # spin_trajectory = spin_trajectory[:1000, :]
    B, N = config_trajectory.shape

    # B and N!
    # now I gotta do some thing, average over all th
    # maybe I want to make a histogram of stuff?
    # delays = np.arange(0, B, dtype=int)
    # print(delays[0:10])
    # q_t = np.zeros(N)
    q_t = np.zeros(B)
    N = 1
    q_man = man_autocorr(config_trajectory)
    for i in range(0, N):
        spin_trajectory = config_trajectory[:, i]
        # auto_corr = np.correlate(spin_trajectory, spin_trajectory, mode='full')
        auto_corr = signal.correlate(spin_trajectory, spin_trajectory, mode='full')
        auto_corr = auto_corr[auto_corr.size // 2:]
        q_t += auto_corr
    q_t = q_t / N
    q_t /= np.max(q_t)
    # lnqt = np.log(q_t)
    # auto_corr = np.corrcoef(spin_trajectory, spin_trajectory)
    avrged_overlap = mq(config_trajectory)
    print(avrged_overlap)
    fig, ax = plt.subplots(2, 1)
    ax = ax.ravel()
    ax[0].plot(q_t, ls='none')
    ax[0].set(xlabel='t', title='q(t)')
    # ax[1].plot(lnqt, ls='none')
    ax[1].semilogy(q_t, ls='none')
    ax[0].set(xlabel='t', title='ln(q(t))')
    # plt.plot(auto_corr / np.max(auto_corr), ls='none')
    # plt.show()
    '''
    start_time = 0
    spin_id = 0
    qs = np.zeros((B, N))
    # not 100% sure how to do this yet!
    start_times = np.arange(0, 1000)
    q_array = np.empty((B, B))
    q_array[:, :] = np.nan
    for i, start_time in enumerate(start_times):
        for j, delay in enumerate(delays):
            q = (
                spin_trajectory[start_time, :] *
                spin_trajectory[delay, :])
            qs[j, :] = q
        q_array[i, :] = np.mean(qs, axis=1)
    print(qs.shape)
    # I'm doing something badly here
    test = q_array[0: 100, 0: 3]
    print(test)
    print(test.shape)
    print(np.nanmean(test, axis=1))
    print(np.nanmean(test, axis=0))
    print(q_array.shape)
    '''
    # plt.plot(np.nanmean(test, axis=1), marker='.')
    # plt.show()
    # result = numpy.correlate(x, x, mode='full')
    # return result[result.size/2:]


def animate_trajectory(trajectory):
    # run_dir = run_dirs[0]
    # md = get_metadata(run_dir)
    # Ts = np.array(md["SweepParameterValues"])
    # print(Ts)
    # fname = run_dir + 'c{}r0trajectory.npz'.format(Tselect)
    # with open(fname, 'rb') as fin:
    #     traj = np.load(fin)
    #     config_traj = traj['prod_traj']
    B, N = trajectory.shape
    L = int(np.sqrt(N))
    snapshots = [trajectory[t].reshape((L, L)) for t in range(0, B)]
    snapshots = np.array(snapshots)
    # fuck need to be carefull cause this is a really long video
    # right?
    # yeah before it had a 5 sec limit, now it just goes on forever!
    # yep keep it limited from now on!
    print(snapshots.shape)
    snapshots = snapshots[-500:]
    animate2Dtrajectory(snapshots)


def animate2Dtrajectory(snapshots):
    frames, _, _ = snapshots.shape
    fps = 30
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # I like to position my colorbars this way, but you don't have to
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    cv0 = snapshots[0]
    im = ax.imshow(cv0)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')

    def animate(i):
        arr = snapshots[i]
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_array(arr)
        im.set_clim(vmin, vmax)
        tx.set_text('Frame {0}'.format(i))
        # In this version you don't have to do anything to the colorbar,
        # it updates itself when the mappable it watches (im) changes
        # return [im, tx]

    ani = animation.FuncAnimation(
        fig, animate, frames=frames,
        interval=1000 / fps)
    ani.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close()


# --------------------------------------------------------------------------- #
def compression_ratio(trajectory):
    B, N = trajectory.shape
    rng = np.random.default_rng()
    trajectory = trajectory.ravel()
    # print(B * N, trajectory.size)
    L = B * N
    bytes_in = bytes(trajectory)
    bytes_out = lzma.compress(bytes_in)
    L_compressed = len(bytes_out)
    # print(L_uncompressed, L_compressed)
    CID = L_compressed / L

    rng.shuffle(trajectory)
    bytes_in = bytes(trajectory)
    bytes_out = lzma.compress(bytes_in)
    L_shuff = len(bytes_out)
    CID_suff = L_shuff / L
    Q = 1 - (CID / CID_suff)

    # print(L_compressed, L_shuff, L, CID, CID_suff)
    # print(Q)
    return Q


def calculate_observables(
        trajectory, true_model, inferred_model, parameters,
        obs_kwrds, data_group):
    output = ()
    if 'basic' in obs_kwrds:
        obs = Observables(trajectory, parameters['T'])
        chi_ferro, chi_sg = obs.compute_spin_spin_correlation()
        m_abs = abs(obs.m)
        q = obs.q
        chi_F = chi_ferro
        # using different definition of chiF now!
        # _, chi_F = obs.chis
        chi_SG = chi_sg
        error, _, _ = planalysis.recon_error_nguyen(
                true_model, inferred_model)
        output = (m_abs, q, chi_F, chi_SG, error)
    if 'e' in obs_kwrds and len(obs_kwrds) == 1:
        error = planalysis.recon_error_nguyen(
                true_model, inferred_model)
        output = output + (error)

    if 'tau' in obs_kwrds:
        autoCorr, tau = overlap_new(trajectory)
        output = output + (tau, autoCorr)
    # print(len(output))
    return output


# maybe here's where I want to use the check exists wrapper?
# I might have to do some sort of better saving for the ISFs
# cause I reckon these are gonna start using a lot of memory!
def h5pysave_dataset(ds_base, ds_name, dataset):
    dset = ds_base.require_dataset(
        ds_name,
        shape=dataset.shape,
        dtype=dataset.dtype,
        compression="gzip")
    dset[()] = dataset


def read_trajectories(data_group, run_dir, obs_kwrds):
    infiles = get_infiles(run_dir)  # [420:]
    observableNames = []

    if 'basic' in obs_kwrds:
        observableNames = ['m', 'q', 'chiF', 'chiSG', 'e']
        for kwrd in obs_kwrds[1:]:
            observableNames.append(kwrd)
    else:
        for kwrd in obs_kwrds:
            observableNames.append(kwrd)

    if len(obs_kwrds) == 1 and 'e' in obs_kwrds:
        observableNames.append('numerator')
        observableNames.append('denominator')
        dt_observables = np.dtype(
            {
                'names': observableNames,
                'formats': [(float)]*len(observableNames)
                })
        observableArray = np.zeros((infiles.size), dtype=dt_observables)

    else:
        dt_observables = np.dtype(
            {
                'names': observableNames,
                'formats': [(float)]*len(observableNames)
                })
        observableArray = np.zeros((infiles.size), dtype=dt_observables)

    parameterNames = ['T', 'h', 'J', 'Jstd', 'cycles']
    dt_parameters = np.dtype(
        {
            'names': parameterNames,
            'formats': [(float)]*len(parameterNames)
            })
    parameterArray = np.zeros((infiles.size), dtype=dt_parameters)
    autocorrs = []
    print('----')
    print("Dataset: {}\nCalculating observables...".format(run_dir))
    print(observableNames)
    for i, infile in enumerate(tqdm(infiles)):
        with h5py.File(infile, 'r') as fin:
            # print(infile)
            traj_dataset = fin['configurations']
            md = dict(traj_dataset.attrs.items())
            if "eq_cycles" in md:
                discard = int(md['eq_cycles'] / md['cycle_dumpfreq'])
                trajectory = fin['configurations'][discard:, :]
            else:
                trajectory = fin['configurations'][()]
            true_model = fin['InputModel'][()]
            inferred_model = fin['InferredModel'][()]

            # print(md['cycle_dumpfreq'], trajectory.shape)
            # parameters = tuple(md[parameter] for parameter in parameterNames)
            parameters = (
                md['T'], md['h'], md['J'], md['Jstd'], md['cycle_dumpfreq'])
            parameterArray[i] = parameters
            # print(output)
            # true_model = true_model * md['T']
            observables = calculate_observables(
                trajectory,
                true_model,
                inferred_model,
                parameterArray[i], obs_kwrds, data_group=data_group)
            if 'tau' in obs_kwrds:
                observableArray[i] = observables[0:-1]
                autocorrs.append(observables[-1])
            else:
                observableArray[i] = observables

    print('--- Saving observables to hdf5 ---')
    h5pysave_dataset(data_group, "parameters", parameterArray)
    h5pysave_dataset(data_group, "observables", observableArray)
    if 'tau' in obs_kwrds:
        autocorrs = np.array(autocorrs)
        h5pysave_dataset(data_group, "autocorr", autocorrs)


# --------------------------------------------------------------------------- #
# HDF5 class for stuff! i.e. read and save and make phase diagrams?
# indicides, thing I want a thing that has 21 x 21 indicies,
# and gives me the values at these points?
# if N is 1 then don't calculate averages, if less than one then do?
# yikes, I probably need to revamp this and save it as a hdf5. Sadge.. 
# lets make two functions, a calculate and a load! that makes more sense :)!
class PhaseDiagram:
    def __init__(self, run_dirs, label_suffix=None):

        print(run_dirs)
        self.run_dirs = run_dirs
        label = self.run_dirs[0].split('_')[0]
        # label = label + '-ACobservables.hdf5'
        if label_suffix is None:
            label_suffix = '-observables.hdf5'
        label = label + label_suffix
        self.label = label
        print(self.label)
        # self.recalculate = recalculate

    # obs_kwrds always has to start with basic
    def calculate(self, obs_kwrds=['basic']):
        # print(self.run_dirs)
        with h5py.File(self.label, "a") as fout:
            # print(fout)
            for repeatID, run_dir in enumerate(self.run_dirs):
                data_group = fout.require_group(run_dir)
                # print(data_group)
                read_trajectories(data_group, run_dir, obs_kwrds)

    def load_run(self, runID):
        with h5py.File(self.label, "r") as fin:
            group = fin[self.run_dirs[runID]]
            print(group)
            datasets = group.keys()
            output = []
            for dataset in datasets:
                print("Found dataset: {}".format(dataset))
                output.append(group[dataset][()])
            # for dataset in datasets:
            # parameters = fin[self.run_dirs[runID]]["parameters"][()]
            # observables = fin[self.run_dirs[ru nID]]["observables"][()]
        output = tuple(output)
        return output

    def averages(self):
        for runID in range(0, len(self.run_dirs)):
            input = self.load_run(runID)
            params = input[0]
            obs = input[1]
            observableNames = obs.dtype.names

            print(params.shape, obs.shape)
            if runID == 0:
                means = np.zeros_like(obs)
            for obsName in observableNames:
                means[obsName] += obs[obsName]
        for obsName in observableNames:
            means[obsName] /= len(self.run_dirs)
        output = (params, means)
        return output

    def load(self):
        if self.recalculate is True:
            print("FORCING RECALCULATION")
        if self.label in glob.glob(self.label) and self.recalculate is False:
            print("LABEL: {} FOUND".format(self.label))
            print("Loading data...")
            npz_file = np.load(self.label)
            observables = npz_file["observables"]
            parameters = npz_file["parameters"]
        else:
            print("Starting calculation...")
            parameters, observables = calc_observables(
                self.run_dirs, self.label)
        self.parameters = parameters
        self.observables = observables

    def calc_averages(self):
        # print(self.parameters.shape)
        # print(self.observables.shape)
        obs_avrg = np.mean(self.observables, axis=1)
        obs_std = np.std(self.observables, axis=1)
        # print(obs_avrg.shape)
        return obs_avrg, obs_std

    def calc_overlap(self):
        label = 'ISF'
        ISF_file = calc_ISFs(
            self.run_dirs, label=label, recalculate=self.recalculate)
        print(ISF_file.files)
        ISFs = ISF_file[label]
        taus = ISF_file['tau']
        Ts = ISF_file['Ts']
        Js = ISF_file['Js']

        # test = [1]
        # with open(label + ".npz", 'wb') as fa:
        #     np.savez(fa, test=test)
        #     print(fa.fileno())
        #     fsz = os.fstat(fa.fileno()).st_size
        #     print(fsz)
        #     print(fa.tell())
        #     # fsz = os.fstat(f.fileno()).st_size
        #     # out = np.load(f)
        #     # while f.tell() < fsz:
        #     #     out = np.vstack((out, np.load(f)))

        #     # np.savez(fappend, ISF=ISFs, tau=taus, Ts=Ts, Js=Js)
        # npz_file = np.load(label + '.npz')
        # print(npz_file.files)
        return ISFs, taus, Ts, Js


# there's got to be a better way to do this!
# helper functions
def set_dimensions(
        phase_diagram_data,
        i_Jmin=0,
        i_Jmax=None,
        i_Tmin=0,
        i_Tmax=None):

    L_sqr = int(np.sqrt(phase_diagram_data.size))
    data = np.copy(phase_diagram_data)
    data = data.reshape(L_sqr, L_sqr, order='C')
    if i_Jmax is None and i_Tmax is None:
        i_Jmax = L_sqr
        i_Tmax = L_sqr
    data = data[i_Jmin: i_Jmax, i_Tmin: i_Tmax]
    return data


def check_npz(function):
    def check_savefile(*args, **kawgs):
        label = kawgs['label']
        recalc = kawgs['recalculate']
        print(label)
        if (label + '.npz') in glob.glob(label + '.npz') and recalc is False:
            print("LABEL: {} FOUND".format(label))
            print("Loading data...")
        else:
            print("Calculating observables")
            function(*args, **kawgs)
        npz_file = np.load(label + '.npz')
        return npz_file
    return check_savefile


def overlap_new(trajectory):
    # print(trajectory.shape)
    B, N = trajectory.shape
    auto_corr = 0
    for i in range(0, N):
        data = trajectory[:, i]
        # normalising or z-scoring it!
        # EXP_si = np.mean(data)
        # STD_si = np.std(data)
        # data = (data - np.mean(data))  # / STD_si
        ac = signal.correlate(
            data, data, mode='full', method='auto')

        ac = ac[int(ac.size/2):]
        auto_corr += ac
    auto_corr /= N
    lags = np.arange(0, auto_corr.size)
    auto_corr_norm = auto_corr/auto_corr.max()
    ac_spline = UnivariateSpline(
        lags, auto_corr_norm-np.exp(-1), s=0)
    ac_roots = ac_spline.roots()
    correlation_time = ac_roots[0]
    # auto_corr /= auto_corr.max()
    return auto_corr, correlation_time


@check_npz
def calc_ISFs(run_dirs, label=None, recalculate=False):
    nReps = len(run_dirs)
    # repISFs = np.
    # repTaus = []
    for repeatID, run_dir in enumerate(run_dirs):
        print(run_dir)
        infiles = get_infiles(run_dir)
        nStatepoints = infiles.size
        spanRange = int(np.sqrt(nStatepoints))
        c = 0
        span_i = 1  # spanRange
        span_j = 3  # spanRange  # this doesn't work!
        Ts = []
        Js = []
        ISFs = []
        taus = []
        # observables = np.zeros((6, nRepeats, spanRange, spanRange))
        for i in range(0, span_i):
            for j in range(0, span_j):
                print(infiles[c])
                with h5py.File(infiles[c], 'r') as f:
                    traj_dataset = f['configurations']
                    metadata = dict(traj_dataset.attrs.items())
                    if "eq_cycles" in metadata:
                        discard = int(
                            metadata['eq_cycles'] /
                            metadata['cycle_dumpfreq'])
                        traj = f['configurations'][discard:, :]
                    else:
                        traj = f['configurations'][()]
                    T = metadata['T']
                    J = metadata['J']
                if repeatID == 0 and i == 0 and j == 0:
                    print(repeatID, i, j)
                    B, N = traj.shape
                    repISF = np.array((nReps, span_i * span_j, B))
                # ISF = overlap_trajectory_fast(traj)
                print(repISF.shape)
                ISF, tau = overlap_new(traj)
                ISFs.append(ISF)
                taus.append(tau)
                if repeatID == 0:
                    Ts.append(T)
                    Js.append(J)
                c = c + 1
        ISFs = np.array(ISFs)
        taus = np.array(taus)

    # repISFs.append(ISFs)
    # repTaus.append(repTaus)
    # repISFs = np.array(repISFs)
    # repTaus = np.array(repTaus)
    print(ISFs.shape, taus.shape)
    Ts = np.array(Ts)
    Js = np.array(Js)
    with open(label + '.npz', 'wb') as fout:
        np.savez(fout, ISF=ISFs, tau=taus, Ts=Ts, Js=Js)


def get_infiles(run_dir):
    # print(run_dir, repeatID)
    file_pattern = join(run_dir, '*.hdf5')
    infiles = sorted(glob.glob(file_pattern))
    infiles = sorted(
        infiles,
        key=lambda x: float(x.split('-J_')[1].split('-Jstd_')[0]))
    return np.array(infiles)


def calc_observables(run_dirs, label):
    nRepeats = len(run_dirs)
    for repeatID, run_dir in enumerate(run_dirs):
        # print(run_dir, repeatID)
        # file_pattern = join(run_dir, '*.hdf5')
        # infiles = sorted(glob.glob(file_pattern))
        # infiles = sorted(
        #     infiles,
        #     key=lambda x: float(x.split('-J_')[1].split('-Jstd_')[0]))

        # infiles = np.array(infiles)
        infiles = get_infiles(run_dir)
        if repeatID == 0:
            nStatepoints = infiles.size
            spanRange = int(np.sqrt(nStatepoints))
            parameters = np.zeros((4, spanRange, spanRange))
            observables = np.zeros((6, nRepeats, spanRange, spanRange))
            print(parameters.shape)
            print(observables.shape)

        print("Dataset: {}\nLoading data...".format(run_dir))
        c = 0
        for i in range(0, spanRange):
            for j in range(0, spanRange):
                with h5py.File(infiles[c], 'r') as f:
                    print("{} / {}".format(c, nStatepoints), end='\r')
                    traj_dataset = f['configurations']
                    metadata = dict(traj_dataset.attrs.items())
                    if repeatID == 0:
                        # print(metadata['T'])
                        # parameters[0, i, j] = metadata['T']
                        # parameters[1, i, j] = metadata['h']
                        # parameters[2, i, j] = metadata['J']
                        # parameters[3, i, j] = metadata['Jstd']
                        parameters[0, j, i] = metadata['T']
                        parameters[1, j, i] = metadata['h']
                        parameters[2, j, i] = metadata['J']
                        parameters[3, j, i] = metadata['Jstd']

                    if "eq_cycles" in metadata:
                        # eq_cycles = metadata['eq_cycles']
                        # cycle_dumpfreq = metadata['cycle_dumpfreq']
                        discard = int(
                            metadata['eq_cycles'] /
                            metadata['cycle_dumpfreq'])
                        traj = f['configurations'][discard:, :]
                    else:
                        traj = f['configurations'][()]
                    true_model = f['InputModel'][()]
                    inferred_model = f['InferredModel'][()]

                obs = Observables(traj, parameters[0, j, i])
                chi_ferro, chi_sg = obs.compute_spin_spin_correlation()
                # print(i, j, repeatID)
                observables[0, repeatID, j, i] = abs(obs.m)
                observables[1, repeatID, j, i] = obs.q

                observables[2, repeatID, j, i] = chi_ferro
                observables[3, repeatID, j, i] = chi_sg

                # observables[4, repeatID, j, i] = planalysis.recon_error(
                #         true_model, inferred_model)
                observables[4, repeatID, j, i] = planalysis.recon_error_nguyen(
                        true_model, inferred_model)

                observables[5, repeatID, j, i] = compression_ratio(traj)

                c = c + 1
    np.savez(label, observables=observables, parameters=parameters)
    return parameters, observables


# --------------------------------------------------------------------------- #
# plots e.g. a correlation plot!
def plotCorrelation(ax, data_x, data_y, xlbl, ylbl, **kwargs):
    # print(data_x.shape)
    data_x = data_x.ravel()
    data_y = data_y.ravel()
    r, p = pearsonr(data_x, data_y)
    ax.plot(
        data_x, data_y,
        **kwargs
        )
    ax.set(xlabel=xlbl, ylabel=ylbl)
