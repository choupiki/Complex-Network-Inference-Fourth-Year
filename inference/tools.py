# general use functions that I might like to use anywhere
import numpy as np
# from scipy.linalg.special_matrices import triu
# import matplotlib.pyplot as plt
from os.path import join, expanduser
from time import perf_counter

from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.interpolate import UnivariateSpline
# from scipy.interpolate import RBFInterpolator

from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge

'''Fitting tools'''


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))


# return mu, & sigma + pval of histogrammed data?
# input binned data!
def gaussian_fit(xs, ys, initialisation=np.array([1, 1, 1])):
    optimal_parameters, pcov = curve_fit(gaussian, xs, ys, p0=initialisation)
    yfit = np.array([gaussian(x, *optimal_parameters) for x in xs])
    residuals = ys - yfit
    return yfit, optimal_parameters, residuals
    # return the y values as an array! that's what i want from this!


# k-means function next?! YEP!
def kmeans(data, n=2):
    # data is raw trajectory data -> 1D! (NOT BINNED)
    data = data.reshape(-1, 1)
    label_indicators = np.arange(0, n)

    kmeans = KMeans(n_clusters=n, random_state=0).fit(data)
    means = kmeans.cluster_centers_
    means = means.ravel()

    labels = kmeans.labels_

    split_data = [data[labels == li].ravel() for li in label_indicators]
    split_data = np.array(split_data, dtype=object)
    # N
    N_cluster = [data.size for data in split_data]
    # print(N_cluster)
    # cluster borders?
    cluster_mins = [np.min(data) for data in split_data]
    cluster_maxs = [np.max(data) for data in split_data]
    # print(cluster_mins, cluster_maxs)
    # metadata_dictionary
    metadata = {
        "means": means,
        "mins": cluster_mins,
        "maxs": cluster_maxs,
        "Nconstituents": N_cluster
    }
    return split_data, metadata


# do a fit using sk-learns kernel ridge regression expects:
# X of shape (n_samples, n_features)
# Y: array-like of shape (n_samples,)
# features: statepoints
# only 1 feature? |m|?
# samples: repeats
def KKR(x_data, y_data, **kwargs):
    print(x_data.shape, y_data.shape)
    X = x_data.reshape(-1, 1)
    y = y_data
    # kernel = pairwise.
    # kernel{linear, poly, rbf, sigmoid, precomputed}
    # [1e0, 1e-1, 1e-2, 1e-3]
    print(y.min())
    clf = KernelRidge(
        alpha=0.6, kernel='poly', degree=7,
        coef0=y.min()
        )
    clf.fit(X, y)
    X_plot = np.linspace(x_data.min(), x_data.max(), 10000)
    y_kr = clf.predict(X_plot.reshape(-1, 1))
    return X_plot, y_kr


def odd_even_check(number):
    if number % 2:
        odd = True  # pass # Odd
    else:
        odd = False  # pass # Even
    return odd


# binned ftt of multiple time series of the same length
# should be shape (Number of series, Number of time points per series)
# could put this into tools! should be shape(Nseries, Ntimepoints)
def fourier_transform_multi_series(trajectories):
    trajectories = np.array(trajectories)
    Nsamples, Nseries = trajectories.shape

    odd = odd_even_check(Nseries)
    # selecting only +ve frequency terms
    if odd is True:
        Ncut = (Nseries - 1) / 2
    elif odd is False:
        Ncut = int((Nseries / 2) - 1)
    else:
        print('You dun goofed')
        return 0
    frequencies = fftfreq(Nseries)[0: Ncut]

    spectra = np.empty((Nsamples, frequencies.size))
    # spec_histogram = np.zeros_like(frequencies)
    # from scipy.signal import hann as window
    for i, trajectory in enumerate(trajectories):
        # w = window(Nseries)
        spectrum = np.abs(fft(trajectory))[0: Ncut]
        # spectrum = 2.0 / Nseries * spectrum  # this is in example; why?
        # spec_histogram += spectrum
        # why am I including 0? shouldn't it start from 1??!?
        # I DON'T KNOW WHATS GOING ON!
        spectra[i] = spectrum

    # spec_histogram /= Nsamples
    spec_means = np.mean(spectra, axis=0)
    spec_std = np.std(spectra, axis=0)
    return frequencies, spec_means, spec_std


''' Matrix Tools'''


# needs symmetric 2d array (matrix)
# includes the diagonal when k=0!
def triu_flat(matrix, k=1):
    triu_indices = np.triu_indices_from(matrix, k)
    upper_values = matrix[triu_indices]
    return upper_values


def split_diag(matrix):
    diagonal = np.diag(matrix)
    triu = np.triu(matrix, 1)
    tril = np.tril(matrix, -1)
    return diagonal, triu, tril


''' String / IO Tools'''


def sort_pathnames(paths, delimiter, delimiter_instance):
    sort_variables = [
        float(path.split(delimiter)[delimiter_instance]) for path in paths]
    sort_variables = np.array(sort_variables)
    sort_indicies = np.argsort(sort_variables)
    paths = np.array(paths)
    return paths[sort_indicies]


''' Timing Tools '''


def exec_time(repeats, function, **kwargs):
    fname = str(function.__name__)
    t1 = perf_counter()
    for _ in range(0, repeats):
        output = function(**kwargs)
    t2 = perf_counter()
    print(
        'FUNCTION: {}\nEXEC TIME: {:.4}s\nREPS: {}'.format(
            fname, t2 - t1, int(repeats)))
    return output


''' Plotting / Analysis Tools '''


def get_threshold_index(ydata, threshold=0, padding=10):
    if len(ydata.shape) > 1:
        return 0
    padding = int(padding)
    indicies = np.arange(0, ydata.size)
    spline = UnivariateSpline(indicies, ydata - threshold, s=0)
    th_roots = spline.roots()
    if len(th_roots) != 0:
        zero_root = int(round(th_roots[0])) + padding
    else:
        zero_root = None
    return zero_root


def bin_mean_xy(xdata, ydata, bin_edges):
    # print(xdata.shape)
    # print(ydata.shape)
    xdata = xdata.ravel()
    ydata = ydata.ravel()
    y_means = []
    bin_centres = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    for i in range(0, bin_edges.size - 1):
        lower_bound = xdata > bin_edges[i]
        upper_bound = xdata <= bin_edges[i + 1]
        truth_mask = lower_bound * upper_bound
        y_means.append(np.nanmean(ydata[truth_mask]))
    y_means = np.array(y_means)
    # y_means[np.isnan(y_means)] = 0
    # print(bin_centres)
    # print(y_means)
    return bin_centres, y_means


def interpolateRBF(xx, yy, z, nNew_points=500, smoothing=0):
    x_min = xx.min()
    x_max = xx.max()
    y_min = yy.min()
    y_max = yy.max()
    print(x_min, x_max)
    print(y_min, y_max)
    xx_smooth, yy_smooth = np.meshgrid(
        np.linspace(x_min, x_max, nNew_points),
        np.linspace(y_min, y_max, nNew_points))

    print(xx.shape, yy.shape, z.shape)
    xy = np.stack([xx.ravel(), yy.ravel()], -1)
    xy_smooth = np.stack([xx_smooth.ravel(), yy_smooth.ravel()], -1)
    print(xy.shape)

    z_rbf = RBFInterpolator(
        xy, z.ravel(),
        smoothing=smoothing,
        kernel='cubic')
    z_smooth = z_rbf(xy_smooth).reshape(xx_smooth.shape)
    return xx_smooth, yy_smooth, z_smooth


''' Plot SK-model'''


def plot_add_SK_analytical_boundaries(ax, xmin, xmax, ymin, ymax):
    home = expanduser('~')
    local_data_dir = join(home, 'extracted_data/Binder1986_Fig49')
    sg_p_file = join(local_data_dir, 'BinderSK_SGP_boundary.txt')
    f_p_file = join(local_data_dir, 'BinderSK_FP_boundary.txt')
    fdash_f_file = join(local_data_dir, 'BinderSK_FdashF_boundary2.txt')

    sg_p_line = np.loadtxt(sg_p_file)
    f_p_line = np.loadtxt(f_p_file)
    fdash_f_line = np.loadtxt(fdash_f_file)

    sg_p_line[0, 0] = xmin
    f_p_line[1, 0] = xmax
    ff_xs = np.linspace(1, xmax, 50)
    ff_spline = UnivariateSpline(fdash_f_line[:, 0], fdash_f_line[:, 1], s=0)
    ax.plot(
        sg_p_line[:, 0], sg_p_line[:, 1],
        marker=',', ls='-', linewidth=2.0, c='k')
    ax.plot(
        f_p_line[:, 0], f_p_line[:, 1],
        marker=',', ls='-', linewidth=2.0, c='k')
    ax.plot(
        ff_xs, ff_spline(ff_xs),
        marker=',', ls='-', linewidth=2.0, c='k',
        label=r'$N \to \infty$ boundary')
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])


# y dataset, x data needed!
def plot_beautiful_errorbars(ax, xs, ydatasets, xlabel=None, ylabel=None):
    ydatasets = np.array(ydatasets)
    sort_args = np.argsort(xs)
    xs = xs[sort_args]
    # print(ydatasets.shape)
    ydatasets = ydatasets[:, sort_args]

    ymeans = np.mean(ydatasets, axis=0)
    ystds = np.std(ydatasets, axis=0)

    y_errormin = ymeans - ystds
    y_errormin_spline = UnivariateSpline(xs, y_errormin, s=0)

    y_errormax = ymeans + ystds
    y_errormax_spline = UnivariateSpline(xs, y_errormax, s=0)

    ax.plot(xs, ymeans, color='k', marker=',')
    xs = np.linspace(xs.min(), xs.max(), xs.size * 4)
    ax.fill_between(
        xs, y_errormin_spline(xs), y_errormax_spline(xs),
        color='grey', alpha=0.5)


def plot_binned_errorbars(ax, x_flat, y_flat, bins, xlabel=None, ylabel=None):
    bin_centers = []
    y_mean = []
    y_std = []
    print('----')
    for i in range(0, bins.size - 1):
        bin_center = bins[i] + ((bins[i + 1] - bins[i]) / 2)
        lower_bound = x_flat > bins[i]
        upper_bound = x_flat <= bins[i + 1]
        truth_mask = lower_bound * upper_bound

        # xis[truth_mask] = bin_center
        bin_centers.append(bin_center)
        y_mean.append(np.nanmean(y_flat[truth_mask]))
        y_std.append(np.nanstd(y_flat[truth_mask]))

    bin_centers = np.array(bin_centers)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)

    # ydatasets = np.array(ydatasets)
    # sort_args = np.argsort(xs)
    # xs = xs[sort_args]
    # print(ydatasets.shape)
    # ydatasets = ydatasets[:, sort_args]
    # ymeans = np.mean(ydatasets, axis=0)
    # ystds = np.std(ydatasets, axis=0)

    y_errormin = y_mean - y_std
    y_errormin_spline = UnivariateSpline(bin_centers, y_errormin, s=1)

    y_errormax = y_mean + y_std
    y_errormax_spline = UnivariateSpline(bin_centers, y_errormax, s=1)
    ax.errorbar(bin_centers, y_mean, y_std, c='k')
    # ax.plot(bin_centers, y_mean, color='k', marker=',')
    # xs = np.linspace(
    #     bin_centers.min(), bin_centers.max(), bin_centers.size * 4)
    # ax.fill_between(
    #     xs, y_errormin_spline(xs), y_errormax_spline(xs),
    #     color='grey', alpha=0.5)
