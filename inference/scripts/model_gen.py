import os
import argparse
import h5py
import json

# from os.path import join

import inference.core.utils as utils

h5py.get_config().track_order = True
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, required=True)
parser.add_argument('--outdir', nargs='?', required=True)
parser.add_argument('--modtype', required=True)
parser.add_argument('--modlabel', nargs='?')
parser.add_argument('--modparams', nargs='*', type=float, required=True)

args = parser.parse_args()
N = args.N

if args.modtype == 'ISING2D':
    L = N ** 0.5
    if N ** 0.5.is_integer() is False:
        print('Error, please enter a square-rootable number of spins, N')
        exit()
    elif len(args.modparams) != 3:
        print('Error, modparams should be T, h, J')
        exit()
    # print(L)
    T = args.modparams[0]
    h = args.modparams[1]
    J = args.modparams[2]
    # print(T, h, J)
    model = utils.ising_interaction_matrix_2D_PBC2(int(L), T, h, J)
elif args.modtype == 'SK':
    if len(args.modparams) != 4:
        print('Error, modparams should be T, h, J0, Jstd')
        exit()
    T = args.modparams[0]
    h = args.modparams[1]
    J0 = args.modparams[2]
    Jstd = args.modparams[3]
    model = utils.SK_interaction_matrix(N, T, h, J0, Jstd)

# this is excessieve, just make datasets with reasonable names!
# save T and J and N and stuff in the metadata?
# if args.modlabel is None:
#     args.modlabel = '{}'.format(
#         args.modtype) + '{}'.format(len(f["InputModels"].keys()))
# this should be done in my other code I think!
os.makedirs(args.outdir, exist_ok=True)
h5py_fname = os.path.join(args.outdir, args.modlabel)
h5py_fname = h5py_fname + '.hdf5'

with h5py.File(h5py_fname, "a") as f:
    # data_group = f.require_group(args.modlabel)
    print('Saving InputModel: {}'.format(args.modlabel))
    ds = f.create_dataset(
        "InputModel", data=model, compression="gzip")
    for key, val in vars(args).items():
        ds.attrs[key] = val

metadata = {args.modlabel: vars(args)}
json_fname = os.path.join(args.outdir, 'InputModels.json')
with open(json_fname, 'a') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4, sort_keys=False)
