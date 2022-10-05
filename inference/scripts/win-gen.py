import os
import argparse

import winhelpers as whelp


parser = argparse.ArgumentParser()
# parser.add_argument('--m', type=str, required=True)
parser.add_argument('--N', type=int, required=True)
parser.add_argument('--n', type=int, required=True)
parser.add_argument('--T', type=float, nargs='+', required=True)
parser.add_argument('--h', type=float, nargs='+', required=True)
parser.add_argument('--J', type=float, nargs='+', required=True)
parser.add_argument('--j', type=float, nargs='+', required=True)
parser.add_argument('--outdir', nargs='?', required=True)
# parser.add_argument('--modtype', required=True)
# parser.add_argument('--modlabel', nargs='?')
# parser.add_argument('--modparams', nargs='*', type=float, required=True)

args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

print(args)
print("Ts: {}".format(args.T))
print("hs: {}".format(args.h))
print("J0s: {}".format(args.J))
print("Jstds: {}".format(args.j))

T_range = whelp.param_range(args.T, args.n)
h_range = whelp.param_range(args.h, args.n)
J0_range = whelp.param_range(args.J, args.n)
Jstd_range = whelp.param_range(args.j, args.n)
print(T_range)
print(h_range)
print(J0_range)
print(Jstd_range)
# this is the bit that I think I want to try and do in parallel!
# maybe this makes a list of things to then give to parallel?
parallel_args = []
for T in T_range:
    for h in h_range:
        for J0 in J0_range:
            for Jstd in Jstd_range:
                label = "T_{:.3f}-h_{:.3f}-J_{:.3f}-Jstd_{:.3f}".format(
                    T, h, J0, Jstd)
                print(label)
                parallel_args.append(
                    [T, h, J0, Jstd, label, args.N, args.outdir])

# this is the parallel bit!
for arg in parallel_args:
    whelp.genSK(arg)
