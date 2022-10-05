import os
import argparse

from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from time import perf_counter
from tqdm import tqdm
import winhelpers_SF as whelp

# e.g. running: python win-exec.py --N 200 --n 2 --T 1 1.4 --h 0 --type 1 --j 1 --mcc 1e5 (<-eq. for this) 5e5 (<- run for) 1e2 (<- save cycles) --outdir test_tf

parser = argparse.ArgumentParser()                              # argparse => allows cmd line passing
parser.add_argument('--N', type=int, required=True)
parser.add_argument('--n', type=int, required=True)
parser.add_argument('--T', type=float, nargs='+', required=True)
parser.add_argument('--h', type=float, nargs='+', required=True)
parser.add_argument('--type', type=int, required=True)
parser.add_argument('--mcc', type=float, nargs='+', required=True)
parser.add_argument('--outdir', nargs='?', required=True)
parser.add_argument('--nJobs', type=int, required=True)
args = parser.parse_args()

if len(args.mcc) != 3:
    print("mcc needs nEQcycles nPRODcycles CycleFreq")          # check
    exit()
os.makedirs(args.outdir, exist_ok=True)

if args.type == 1:
    graph = 'Complete'
if args.type == 2:
    graph = '2D Ising'

print(args)
print("Ts: {}".format(args.T))
print("hs: {}".format(args.h))
print(graph)
print('n\n\n', args.outdir)

T_range = whelp.param_range(args.T, args.n)
h_range = whelp.param_range(args.h, args.n)

print('Preparing Commands...')
parallel_args = []                               
for T in T_range:                                             # allows for individual diff. models to be sampled
    for h in h_range:
        label = "T_{:.3f}-h_{:.3f}-Type_{:}".format(
            T, h, graph)
        # print(label)
        parallel_args.append(
            [T, h, graph, label, args.N, args.outdir, args.n])
        
if args.type == 1:
    print('Saving Models')
    nJobs = args.nJobs
    t0 = perf_counter()                                           # generating models
    fnames = (Parallel(n_jobs=nJobs)(                             # parallel core usage, just like a for loop
        delayed(whelp.genCM)(arg) for arg in tqdm(parallel_args)  # give fn to delayed, input args
        ))
    get_reusable_executor().shutdown(wait=True)
    t1 = perf_counter()
    print('Time taken: {}s'.format(t1 - t0))

if args.type == 2:
    print('Saving Models')
    nJobs = args.nJobs
    t0 = perf_counter()                                           # generating models
    fnames = (Parallel(n_jobs=nJobs)(                             # parallel core usage, just like a for loop
        delayed(whelp.gen2D)(arg) for arg in tqdm(parallel_args)  # give fn to delayed, input args
        ))
    get_reusable_executor().shutdown(wait=True)
    t1 = perf_counter()
    print('Time taken: {}s'.format(t1 - t0))


print('Running Monte-Carlo')
args.mcc = [int(a) for a in args.mcc]
print(args.mcc)
t0 = perf_counter()
Parallel(n_jobs=nJobs)(
    delayed(whelp.mccCM_2D)(fname, args.mcc) for fname in tqdm(fnames)
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
