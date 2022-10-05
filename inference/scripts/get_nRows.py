import argparse
import h5py

parser = argparse.ArgumentParser()

parser.add_argument('--file', nargs='?', required=True)
args = parser.parse_args()

with h5py.File(args.file, 'r') as f:
    nSamples, nFeatures = f['configurations'].shape
    print(nFeatures)
