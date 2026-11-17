from __future__ import annotations
from model import HubbardModel
import numpy as np
from scipy.linalg import expm

"""
In this file we will define the monte carlo step of the auxiliary field quantum monte carlo algorithm.
We will propose a new configuration of the auxiliary field X, by flipping a single element of the auxiliary field X,
and accept or reject it based on the weight of the new configuration.
"""

class MonteCarlo:
    """
    This class will contain the monte carlo step of the auxiliary field quantum monte carlo algorithm.
    """
    def __init__(self, M, beta, X, model: HubbardModel):
        self.model = model
        self.M = M
        self.N = model.L * model.L
        self.beta = beta
        self.dtau = beta / M
        self.X = X
        self.T = self.T()
        self.lam = np.arccosh(np.exp(self.dtau * model.U / 2))
        self.current_weight, self._current_P_up, self._current_P_down = self.weight(self.X)

    def flip(self):
        """
        Perform a single monte carlo flip over the auxiliary field X.
        We will propose a new configuration of the auxiliary field X, by flipping a single element of the auxiliary field X,
        and accept or reject it based on the weight of the new configuration. 
        The flip function will do one flip.
        """
        flipped = False
        
        # Propose a new configuration by sekecting a random element of the auxiliary field X and flipping it
        l = np.random.randint(0, self.M)
        i = np.random.randint(0, self.N)
        self.X[l, i] *= -1

        # Calculate the weight of the new configuration
        weight_new, P_up_new, P_down_new = self.weight(self.X)


        # Accept or reject the new configuration based on the Metropolis algorithm
        if np.random.rand() < min(1, weight_new / self.current_weight):
            # Accept the new configuration
            self.current_weight = weight_new
            self._current_P_up = P_up_new
            self._current_P_down = P_down_new
            flipped = True
        else:
            # Reject the new configuration and flip the element back
            self.X[l, i] *= -1
        return flipped

    def T(self):
        """
        Return the half exponential matrix of the hopping matrix in the symmetric Trotter approximation.
        Since the hopping matrix is non diagonal, this is non trivial. We will therefore use scipy.linalg 
        function expm to calculate it. Since the hopping matrix is a constant matrix, we can calculate it once and store it for later use.
        """
        k = self.model.hopping_matrix()
        return expm(-0.5 * self.dtau * k)

    def V_l_sigma(self, X, l, sigma):
        """
        Return the exponential matrix of the repulsive term at imaginary time slice l after HS transformation 
        with depedence on the Auxiliery field X. Since D_l is already diagonal, taking the exponential is simple.
        """         
        return np.diag(np.exp(sigma * self.lam * X[l]))  

    def B_l_sigma(self, X, l, sigma):
        V = self.V_l_sigma(X, l, sigma)
        return self.T @ V @ self.T

    def P_l_sigma(self, X, l, sigma):
        """
        Return the product of B matrices from 0 to l for a given spin sigma.
        """
        P = np.eye(self.T.shape[0])
        for i in range(l + 1):
            P = P @ self.B_l_sigma(X, i, sigma)
        return P

    def weight(self, X):
        """
        Return the weight of a given configuration of the auxiliary field X.
        The weight is given by the product of the determinants of (I + P) for both spin up and spin down, 
        where P is the product of B matrices from 0 to M-1 for a given spin sigma.
        We will also return the P_up and P_down matrices for later use in calculating the Green's function.
        """
        P_up = self.P_l_sigma(X, self.M - 1, 1)
        P_down = self.P_l_sigma(X, self.M - 1, -1)
        weight = np.linalg.det(np.eye(P_up.shape[0]) + P_up) * np.linalg.det(np.eye(P_down.shape[0]) + P_down)
        return weight, P_up, P_down
