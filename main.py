import numpy as np
from scipy.linalg import expm


"""
Our Hubbard Model Hamiltonian contains an hopping term K and a repulsive term V. 
The repulsive term after HS transformation will give rise to a diagonal matrix dependent
on the auxiliary field X. The auxiliary field X have dimensions M x L^2, where M
is the number of imaginary time slices that we have M = beta / Delta_tau.
"""

U = 4.0
beta = 1
M = 10
dtau = beta / M 
lam = np.arccosh(np.exp(dtau * U / 2))


def hopping_matrix(L, t = float(1.0)):
    """
    Let i and j correspond to lattice sites, not lattice coordinates. 
    Then for the single particle matrix we have that k_ij = -t if i and j
    are nearest neighbours and zero otherwise, where k is a N_s x N_s matrix
    with N_s = L^2 for a square lattice. Since the number of rows and columns of
    k is N_s, it is easiest if we enumerate the 2d lattice sites.
    We will use the enumeration i = x + yL.
    """
    N = L * L
    k = np.zeros((N, N), dtype=float) 

    for i in range(N):
        x1 = i % L
        y1 = i // L
        for j in range(N):
            x2 = j % L
            y2 = j // L
            if y1 == y2 and abs(x1 - x2) in {1, L-1}:
                k[i][j] = -t
            elif x1 == x2 and abs(y1 - y2) in {1, L-1}:
                k[i][j] = -t
            else:
                k[i][j] = 0
    return k

def T(L):
    """
    Return the half exponential matrix of the hopping matrix in the symmetric Trotter approximation.
    Since the hopping matrix is non diagonal, this is non trivial. We will therefore use scipy.linalg 
    function expm to calculate it.
    """
    k = hopping_matrix(L)
    return expm(-0.5 * dtau * k)

def V_l_sigma(X, l, sigma):
    """
    Return the exponential matrix of the repulsive term at imaginary time slice l after HS transformation 
    with depedence on the Auxiliery field X. Since D_l is already diagonal, taking the exponential is simple.
    """         
    return np.diag(np.exp(sigma * lam * X[l]))  

def B_l_sigma(X, l, sigma, T):
    V = V_l_sigma(X, l, sigma)
    return T @ V @ T
