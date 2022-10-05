import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

def SW_matrix(N, k, prob, h, T):
    G = nx.generators.random_graphs.newman_watts_strogatz_graph(N, k, prob)
    J = nx.convert_matrix.to_numpy_array(G)
    np.fill_diagonal(J, h)
    J = J / T
    return J, G

def SF_matrix(N, gamma, h, T):
    seq = nx.utils.random_sequence.powerlaw_sequence(N, gamma)
    G = nx.expected_degree_graph(seq, selfloops=False)
    J = nx.to_numpy_array(G)
    np.fill_diagonal(J, h)
    J = J / T
    return J, G

def complete_matrix(N, h, T):
    G = nx.complete_graph(N)
    J = nx.to_numpy_array(G)
    np.fill_diagonal(J, h)
    J = J / T
    return J, G

def twoD_ising_matrix(N, h, T):
    G = nx.generators.lattice.grid_2d_graph(N, N, periodic=False)
    J = nx.to_numpy_array(G)
    np.fill_diagonal(J, h)
    J = J / T
    return J, G

