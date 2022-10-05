import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
from datetime import timedelta

import inference.analysis.new as analysis


# FOR 1E4 1E4 10 IT TAKES
# times vary on bc, let's use max and round up?
# BC 4.735892880707979 / 5.362952798604965 / 4.7255641259253025
# let's use 5.5
# LOCAL: 2.503323516 s
# -> Bc = 6 / 2.5 * local =  2.4 * local

# log_quadratic
def log_quadratic(x, a, b, c):
    return (a * x**2) + (b * x) + c


parser = argparse.ArgumentParser()
parser.add_argument('--rundirs', nargs='*', required=True)
args = parser.parse_args()
plt.style.use('~/Devel/styles/custom.mplstyle')


run_dirs = args.rundirs
observableNames = ['m', 'q', 'chiF', 'chiSG', 'e', 'tau']
observableLabels = [
    r'$|m|$', r'$q$',
    r'$\chi _{F}$', r'$\chi _{SG}$',
    r'$\epsilon$', r'$\tau$']
parameterNames = ['T', 'J']

pd = analysis.PhaseDiagram(run_dirs)
params, obs = pd.averages()
print("----")
eq_cycles = int(5e5)
B = 1e4
# it definitely needs a buffer, cause atm I'm only looking at the MC loop!
buffer = 2 * 60  # buffer in seconds, 5 mins too much?
infiles = analysis.get_infiles(run_dirs[0])

# --- Gaussian Blur & prepare tau_alpha --- #
params = analysis.set_dimensions(params, 0, None, 0, None)
obs = analysis.set_dimensions(obs, 0, None, 0, None)
obs['tau'][obs['tau'] < 1] = 1
obs['tau'] = obs['tau'] * 10
obs['tau'] = ndimage.gaussian_filter(obs['tau'], sigma=1.0, order=0)
# obs['tau'] = obs['tau'].astype(int)
nMCCs_to_sample = np.rint((B * obs['tau']))

# --- Get back into list shape --- #
params = params.reshape(params.size)
obs = obs.reshape(obs.size)
nMCCs_to_sample = nMCCs_to_sample.reshape(params.size)

# --- Get execution times fit --- #
mcc_times = np.loadtxt(
    '/Users/mk14423/Devel/fmri2/executables/mcc_test/simtimes.txt',
    delimiter='\t')
mcc_times = mcc_times[mcc_times[:, 0].argsort()]
mcc_times[:, 1] = mcc_times[:, 1] * (6 / 2.5)
log_times = np.log10(mcc_times)

popt, pcov = curve_fit(log_quadratic, log_times[:, 0], log_times[:, 1])
t_execs = 10 ** log_quadratic(np.log10(nMCCs_to_sample), *popt)
t_execs = np.rint(t_execs)


# --- print parameters for simulation? --- #
with open('commands-mc.txt', 'w') as fout:
    for infile, sampling_freq in zip(infiles, obs['tau']):
        sampling_freq = int(sampling_freq)
        prod_cycles = int(sampling_freq * B)
        print(sampling_freq, prod_cycles / sampling_freq)
        fout.write(
            "python /mnt/storage/home/mk14423/pyscripts/mc_run.py"
            " --modfile {} --mccsample {} {} {}\n".format(
                infile, eq_cycles, prod_cycles, sampling_freq)
        )

with open('commands-times.txt', 'w') as fout:
    for infile, t_exec in zip(infiles, t_execs):
        t_seconds = t_exec + buffer
        if t_seconds >= 60 * 60:
            print("WATCH OUT, LONGER THAN AN HOUR!")
            print(infile)
        real_time = timedelta(seconds=(t_seconds))
        real_time = str(real_time)
        # print(t_exec, real_time)
        fout.write("{} {}\n".format(infile, real_time))

print(t_execs.max(), (obs['tau'] * B).max())
fig, ax = plt.subplots()
ax.plot(nMCCs_to_sample, t_execs, marker='o', ls='none')
ax.set(xlabel=r'$B$', ylabel=r'$t _{exec}$')
ax.set(xscale='log', yscale='log')
plt.show()


x = np.logspace(3, 9, 500)
y = log_quadratic(np.log10(x), *popt)  # y = log(t)
fit = 10 ** y
fig, ax = plt.subplots()
ax.plot(mcc_times[:, 0], mcc_times[:, 1], marker='o', ls='none')
ax.plot(x, fit, marker=',')
ax.set(xlabel=r'$B$', ylabel=r'$t _{exec}$')
ax.set(xscale='log', yscale='log')
plt.show()
# taus = obs['tau']
# taus[taus < 1] = 1
# taus = obs['tau'] * 10  # in units of MCCs to get decorellated configs
# taus = taus.astype(int)
# ooh I've got to gaussian blur in 2D!!!
# taus = ndimage.gaussian_filter(taus, sigma=1.0, order=0)
