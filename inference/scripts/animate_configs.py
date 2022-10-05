import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import data_namer 
from tqdm import tqdm
import winhelpers_SF as whelp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics
import scipy as sp
import math
from apng import APNG

#Folder = 'SF_N1000_Y2-5.1'
#Fname = 'T_0.872-h_0.000-gamma_2.540.hdf5'

def open_plots(folder, fname):
    path = os.path.join(folder, fname)
    with h5py.File(path, 'r') as fin:
        modIn = fin['InputModel'][()]                   # reloading file as np array 
        modOut = fin['InferredModel'][()]
        md = dict(fin['configurations'].attrs.items())
        md_in = dict(fin['InputModel'].attrs.items())
        T = md.get('T')
        N = md.get('N')
        config_traj = np.array(fin.get('configurations'))
        return config_traj, N
    
def plot_PT(folder, fname, i):
    path = os.path.join(folder, fname)
    with h5py.File(path, 'r') as fin:
        md = dict(fin['configurations'].attrs.items())
        eq_cycles = md.get('eq_cycles')
        prod_cycles = md.get('prod_cycles')
        dump = md.get('cycle_dumpfreq')
        N = md.get('N')
        T = md.get('T')
        config_traj = np.array(fin.get('configurations'))
        spin_pic = config_traj[:][int(0.7*((prod_cycles/dump)))]
        new_path = 'C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Presentation/animations/PT/' + files_folder + '/'
        #os.makedirs(new_path)
        plt.imshow(spin_pic.reshape(int(np.sqrt(N)), int(np.sqrt(N))), cmap=plt.get_cmap('binary'))
        plt.savefig(new_path + str(i) + '.png')
        plt.clf()
        return new_path, i
    
    
    
    
    
    

parser = argparse.ArgumentParser()
parser.add_argument('--indir', nargs='?', required=True)
# parser.add_argument('--outdir', nargs='?', required=True)
# parser.add_argument('--nJobs', type=int, required=True)
parser.add_argument('--type', type=int, required=True)
args = parser.parse_args()

# set parsed inputs as vars:
files_folder = args.indir


# For animation of MCC over one state point
if args.type == 1:
    folder = "C:/Users/oscar/devel/fmri2/inference/scripts/{}".format(files_folder)
    files = data_namer.state_point_lister(files_folder)
    fname = files[0][50]
    config_traj, N = open_plots(folder, fname)
    fig = plt.figure(figsize=(int(np.sqrt(N)), int(np.sqrt(N))))
    os.makedirs('C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Presentation/animations/'+ fname)
    for i in range(len(config_traj[:][0])):
        plt.imshow(config_traj[i][:].reshape(int(np.sqrt(N)), int(np.sqrt(N))), cmap=plt.get_cmap('binary'))
        plt.savefig('C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Presentation/animations/'+ fname + '/' + str(i)+'.png')
        plt.clf()
        
    filenames = ['C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Presentation/animations/'+ fname + '/' + str(i)+'.png' for i in range(len(config_traj[:][0]))]
    APNG.from_files(filenames, delay=100).save('C:/Users/oscar/Documents/School Stuff/Physics/4th Year/Research Project/Presentation/animations/'+fname+'-.apng')

# For animation over T values
elif args.type == 2:
    
    # List path to and file names in tuple:
    files = data_namer.state_point_lister(files_folder)
    print(files)
    T_list = []
    for i in range(len(files[0])):
        new_path, T = plot_PT(files[1], files[0][i], i)
        T_list.append(T) 

    filenames = os.listdir(new_path)
    for i in range(len(filenames)):
        filenames[i] = new_path + filenames[i]   
    
    #print(filenames)
    APNG.from_files(filenames, delay=100).save(new_path + files_folder + '-.apng')


