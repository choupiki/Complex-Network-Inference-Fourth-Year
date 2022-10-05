import argparse
import h5py
# import numpy as np
# import matplotlib.pyplot as plt

from inference.sweep import MonteCarlo2

parser = argparse.ArgumentParser()

parser.add_argument('--modfile', nargs='?', required=True)
parser.add_argument('--mccsample', nargs=3, type=str, required=True)

args = parser.parse_args()

args.mccsample = map(float, args.mccsample)
args.mccsample = list(map(int, args.mccsample))
# print(args.modfile)
# I might acutally want the dumpfreq to be float!

with h5py.File(args.modfile, 'r') as fin:
    model_dataset = fin['InputModel']
    model = model_dataset[:]
    modparams = {}
    # print(model_dataset.attrs['modparams'])
    modparams['T'] = model_dataset.attrs['modparams'][0]
    modparams['h'] = model_dataset.attrs['modparams'][1]
    modparams['J'] = model_dataset.attrs['modparams'][2]

    if len(model_dataset.attrs['modparams']) > 3:
        modparams['Jstd'] = model_dataset.attrs['modparams'][3]


ising_sim = MonteCarlo2(model, modparams, args.modfile)
print('----')
ising_sim.setHyperParameters(
    eq_cycles=args.mccsample[0],
    prod_cycles=args.mccsample[1],
    cycle_dumpfreq=args.mccsample[2]
    )

ising_sim.run()
