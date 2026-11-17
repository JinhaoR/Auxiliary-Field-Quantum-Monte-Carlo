from __future__ import annotations
from model import HubbardModel
from monte_carlo import MonteCarlo
import numpy as np
import scipy.linalg
"""
In this runtime file we will create the runtime class that will be used to run the simulation. 
This class will be responsible for initializing the model, running the Monte Carlo simulation, and collecting the results.
Here we will perfome full sweeps over the auxiliary field X, and collect the results of the Green's function and the weight of the configuration.
"""

class Runtime:
    """
    This class will be responsible for running the simulation.
    """
    def __init__(self, model: HubbardModel, M: int, beta: float):
        self.model = model
        self.M = M
        self.N = model.L * model.L
        self.beta = beta

        self.X = self.initialize_auxiliary_field()
        self.MC = MonteCarlo(M, beta, self.X, model)

        self.densitys = [self.density()]
        self.dim = self.M * self.N


    def run(self, thermalization: int, measure: int):
        """
        Run the simulation for a given number of sweeps.
        """
        for _ in range(thermalization):
            self.sweep()
        for _ in range(measure):
            self.sweep()
            self.densitys.append(self.density())

    def sweep(self):
        """
        Perform a single monte carlo sweep over the auxiliary field X.
        We will propose a new configuration of the auxiliary field X, by flipping a single element of the auxiliary field X,
        and accept or reject it based on the weight of the new configuration. 
        The sweep function will do M x N flips.
        """
        for _ in range(self.dim):
            self.MC.flip()

    def density(self):
        """
        Calculate the density of the current configuration of the auxiliary field X.
        """
        G_up, G_down = self.green_function()
        n_up = 1 - np.diag(G_up)
        n_down = 1 - np.diag(G_down)
        return np.mean(n_up + n_down)

    def green_function(self):
        """
        Calculate the Green's function of the current configuration of the auxiliary field X.
        """
        G_up = scipy.linalg.inv(np.eye(self.MC.N) + self.MC._current_P_up)
        G_down = scipy.linalg.inv(np.eye(self.MC.N) + self.MC._current_P_down)
        return G_up, G_down

    def initialize_auxiliary_field(self):
                """
                Initialize the auxiliary field X with random values of +1 or -1.
                """
                X = np.random.choice([-1, 1], size=(self.M, self.N))
                return X