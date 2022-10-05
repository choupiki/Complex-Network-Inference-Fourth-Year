import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import data_namer 

# import inference.analysis.planalysis as planalysis
#from inference import tools

def nicer_hist(ax, data, nbins=50, **kwargs):
    weights = np.ones_like(data) / len(data)
    hist, bin_edges = np.histogram(data, bins=nbins, weights=weights)
    bin_centres = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    ax.plot(bin_centres, hist)
    ax.set(xlabel=r'$x$', ylabel=r'$P(x)$')

files = data_namer.state_point_lister('SF_RE_withT_3')
fname = files[0]

def inf_plots(folder, fname):
    path = os.path.join(folder, fname[i])
    with h5py.File(path, 'r') as fin:
        modIn = fin['InputModel'][()]                   # reloading file as np array 
        modOut = fin['InferredModel'][()]
        md = dict(fin['configurations'].attrs.items())
        mag = np.array(fin.get('configurations')).mean()
    """
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(modIn)
    ax[1].imshow(modOut)
    plt.title(fname[i])
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    plt.show()
    """
    J_in = modIn
    h_in = np.diagonal(modIn)
    J_out = modOut
    h_out = np.diagonal(modOut)

    r_err = np.sqrt(np.sum((np.subtract(J_in, J_out))**2)/np.sum(J_in)**2)
    return r_err, mag

r_err = []
m = []
for i in range(len(fname)):
    r_err.append(inf_plots(files[1], fname)[0])
    m.append(inf_plots(files[1], fname)[1])
    
    

plt.plot(r_err)
plt.show()

plt.plot(m)
plt.show()


#def err_phase_diagram()