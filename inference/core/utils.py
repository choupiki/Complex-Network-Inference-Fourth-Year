import numpy as np
from pathlib import Path

'''
def ising_interaction_matrix_2D_PBC(L, h, jval):
    N = L ** 2
    J = np.zeros((N, N))
    for i in range(0, N):
        if i % L == 0:
            J[i, i+L-1] = jval
        else:
            J[i, i-1] = jval

        if (i+1) % L == 0:
            J[i, i-L+1] = jval
        else:
            J[i, i+1] = jval

        if i < L:
            J[i, i + (N-L)] = jval
        else:
            J[i, i-L] = jval

        if i >= (N-L):
            J[i, i-(N-L)] = jval
        else:
            J[i, i+L] = jval
    np.fill_diagonal(J, h)
    return J
'''


# you feed in T and h and jval get divided by it!
# I dont really see how this will make all that much of a difference...
# dividing it all by T now or later doesnt really mean much?
def ising_interaction_matrix_2D_PBC2(L, T=1, h=0, jval=1):
    N = L ** 2
    J = np.zeros((N, N))
    for i in range(0, N):
        if i % L == 0:
            J[i, i+L-1] = jval / T
        else:
            J[i, i-1] = jval / T

        if (i+1) % L == 0:
            J[i, i-L+1] = jval / T
        else:
            J[i, i+1] = jval / T

        if i < L:
            J[i, i + (N-L)] = jval / T
        else:
            J[i, i-L] = jval / T

        if i >= (N-L):
            J[i, i-(N-L)] = jval / T
        else:
            J[i, i+L] = jval / T
    np.fill_diagonal(J, h / T)
    return J


# lets plot a few and histogram it to see!
# think about how to incorporate T into this again!!!
# need to make sure I normalise this stuff by N!!!!
# so has a mean on Jo/N and a Var J^2 / N
# lets try with 0 std to begin? WIll give error so maybe not?
# check that distrbution is of order 1/N^2!
def SK_interaction_matrix(N, T=1, h=0, jmean=0, jstd=1):
    jmean = jmean / N
    jstd = jstd / np.sqrt(N)
    rand = np.random.normal(loc=jmean, scale=jstd, size=(N, N))
    J = np.tril(rand) + np.tril(rand, -1).T
    np.fill_diagonal(J, h)
    # so its all / T when I return it!!
    J = J / T
    return J


# return 1D object containing list of initial Ising Spins
# (i.e. +/- 1 only -> binary)
def initialise_ising_config(N, option):
    if option == -1:
        config = -np.ones(N)
    elif option == 0:
        config = np.random.randint(2, size=N)
        config[config == 0] = -1
    elif option == 1:
        config = np.ones(N)
    else:
        print('Invalid initialisation choice made')
        return 1
    return config


# this should really be elsewhere!!
def gen_spin_matrix(si, sij):
    spin_matrix = np.copy(sij)
    spin_matrix[np.diag_indices_from(spin_matrix)] = si  # fills diagonal
    return spin_matrix


# don't use this anymore!
# I should really clean up my "codebase"
def set_metadata(
        run_dir, N, pname, pvals,
        cycles_eq=1 * (10 ** 4),
        cycles_prod=5 * (10 ** 4),
        cycle_dumpfreq=10, reps=1):

    Path(run_dir).mkdir(exist_ok=True)
    md = {
        'RunDirectory': run_dir,
        'SystemSize': N,
        'EqCycles': cycles_eq,
        'ProdCycles': cycles_prod,
        'CycleDumpFreq': cycle_dumpfreq,
        'Repetitions': reps,
        'SweepParameterName': pname,
        'SweepParameterValues': pvals.tolist()}
    return md
