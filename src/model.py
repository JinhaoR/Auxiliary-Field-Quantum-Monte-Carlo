import numpy as np

"""
Our Hubbard Model Hamiltonian contains an hopping term K and a repulsive term V. 
The repulsive term after HS transformation will give rise to a diagonal matrix dependent
on the auxiliary field X. The auxiliary field X have dimensions M x L^2, where M
is the number of imaginary time slices that we have M = beta / Delta_tau.

In this file we will define the hopping matrix K, the half exponential of the hopping matrix T,
the exponential of the repulsive term V_l_sigma and the B_l_sigma matrix which is the
product of the half exponential of the hopping matrix and the exponential of the repulsive term.

We will them use these matrices to construct the Green's function and the weight of a given configuration of the auxiliary field X.
"""

class HubbardModel:
    """
    This class will contain the parameters of the Hubbard model,
    and the matrices that we will use to calculate the Green's function
    and the weight of a given configuration of the auxiliary field X.
    """
    def __init__(self, L=8, t = 1.0, U=4):
        self.L = L
        self.t = t
        self.U = U

    def hopping_matrix(self):
        """
        Let i and j correspond to lattice sites, not lattice coordinates. 
        Then for the single particle matrix we have that k_ij = -t if i and j
        are nearest neighbours and zero otherwise, where k is a N_s x N_s matrix
        with N_s = L^2 for a square lattice. Since the number of rows and columns of
        k is N_s, it is easiest if we enumerate the 2d lattice sites.
        We will use the enumeration i = x + yL.
        """
        N = self.L * self.L
        k = np.zeros((N, N), dtype=float) 

        for i in range(N):
            x1 = i % self.L
            y1 = i // self.L
            for j in range(N):
                x2 = j % self.L
                y2 = j // self.L
                if y1 == y2 and abs(x1 - x2) in {1, self.L-1}:
                    k[i][j] = -self.t
                elif x1 == x2 and abs(y1 - y2) in {1, self.L-1}:
                    k[i][j] = -self.t
                else:
                    k[i][j] = 0
        return k
