import os
import argparse

from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from time import perf_counter
from tqdm import tqdm
import winhelpers_SWSF as whelp

# e.g. running: python win-exec.py --N 200 --n 2 --T 1 1.4 --h 0 --J 0.5 1 --j 1 --mcc 1e5 (<-eq. for this) 5e5 (<- run for) 1e2 (<- save cycles) --outdir test_tf

parser = argparse.ArgumentParser()                              # argparse => allows cmd line passing
parser.add_argument('--N', type=int, required=True)
parser.add_argument('--n', type=int, required=True)
parser.add_argument('--T', type=float, nargs='+', required=True)
parser.add_argument('--h', type=float, nargs='+', required=True)
parser.add_argument('--gamma', type=float, nargs='+', required=True)
parser.add_argument('--mcc', type=float, nargs='+', required=True)
parser.add_argument('--outdir', nargs='?', required=True)
parser.add_argument('--nJobs', type=int, required=True)
args = parser.parse_args()

if len(args.mcc) != 3:
    print("mcc needs nEQcycles nPRODcycles CycleFreq")          # check
    exit()
os.makedirs(args.outdir, exist_ok=True)

print(args)
print("Ts: {}".format(args.T))
print("hs: {}".format(args.h))
print("gammas: {}".format(args.gamma))


T_range = whelp.param_range(args.T, args.n)
h_range = whelp.param_range(args.h, args.n)
gamma_range = whelp.param_range(args.gamma, args.n)

print('Preparing Commands...')
parallel_args = []                               
for T in T_range:                                             # allows for individual diff. models to be sampled
    for h in h_range:
        for gamma in gamma_range:
            label = "T_{:.3f}-h_{:.3f}-gamma_{:.3f}".format(
                T, h, gamma)
            # print(label)
            parallel_args.append(
                [T, h, gamma, label, args.N, args.outdir])
print('Saving Models')
nJobs = args.nJobs
t0 = perf_counter()                                           # generating models
fnames = (Parallel(n_jobs=nJobs)(                             # parallel core usage, just like a for loop
    delayed(whelp.genSF)(arg) for arg in tqdm(parallel_args)  # give fn to delayed, input args
    ))
get_reusable_executor().shutdown(wait=True)
t1 = perf_counter()
print('Time taken: {}s'.format(t1 - t0))

print('Running Monte-Carlo')
args.mcc = [int(a) for a in args.mcc]
print(args.mcc)
t0 = perf_counter()
Parallel(n_jobs=nJobs)(
    delayed(whelp.mccSF)(fname, args.mcc) for fname in tqdm(fnames)
    )
get_reusable_executor().shutdown(wait=True)
t1 = perf_counter()
print('Time taken: {}s'.format(t1 - t0))

print('Running Inference')
t0 = perf_counter()
Parallel(n_jobs=nJobs)(
    delayed(whelp.infer)(fname) for fname in tqdm(fnames)
    )
get_reusable_executor().shutdown(wait=True)
t1 = perf_counter()
print('Time taken: {}s'.format(t1 - t0))
