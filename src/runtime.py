from __future__ import annotations
from model import HubbardModel
from monte_carlo import MonteCarloDeterminant, MonteCarloGreen
import numpy as np
import scipy.linalg
"""
In this runtime file we will create the runtime class that will be used to run the simulation. 
This class will be responsible for initializing the model, running the Monte Carlo simulation, and collecting the results.
Here we will perfome full sweeps over the auxiliary field X, and collect the results of the Green's function and the weight of the configuration.
"""

class RuntimeDeterminant:
    """
    This class will be responsible for running the simulation.
    """
    def __init__(self, model: HubbardModel, M: int, beta: float):
        self.model = model
        self.M = M
        self.N = model.L * model.L
        self.beta = beta

        self.X = self.initialize_auxiliary_field()
        self.MC = MonteCarloDeterminant(M, beta, self.X, model)

        self.densitys = []
        self.double_occupancies = []


    def run(self, thermalization: int, measure: int):
        """
        Run the simulation for a given number of sweeps.
        """
        for _ in range(thermalization):
            self.MC.sweep()
        for _ in range(measure):
            self.MC.sweep()
            density, double_occupancy = self.observable()
            self.densitys.append(density)
            self.double_occupancies.append(double_occupancy)

    def observable(self):
        """
        Calculate the density of the current configuration of the auxiliary field X.
        """
        G_up, G_down = self.green_function()
        n_up = 1 - np.diag(G_up)
        n_down = 1 - np.diag(G_down)

        density = np.mean(n_up + n_down)
        double_occupancy = np.mean(n_up * n_down)
        return density, double_occupancy

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

class RuntimeGreenFunction:
    """
    This class will be responsible for running the simulation using
    propagation of the Green's function rather than calculating 
    determinant weights.
    """
    def __init__(self, model: HubbardModel, M: int, beta: float):
        self.model = model
        self.M = M
        self.L = model.L
        self.N = model.L * model.L
        self.beta = beta

        self.X = self.initialize_auxiliary_field()
        self.MC = MonteCarloGreen(M, beta, self.X, model)

        self.densitys = []
        self.double_occupancies = []
        self.spin_correlations = []

        self.eta = np.array([(-1)**(i % self.L + i // self.L) for i in range(self.N)])
        self.spin_structure_factors = []
        self.dim = self.M * self.N


    def run(self, thermalization: int, measure: int):
        """
        Run the simulation for a given number of sweeps.
        """
        for _ in range(thermalization):
            self.MC.sweep()
        for _ in range(measure):
            self.MC.sweep()
            density, double_occupancy, spin_correlation, spin_structure_factors = self.observable()
            self.densitys.append(density)
            self.double_occupancies.append(double_occupancy)
            self.spin_correlations.append(spin_correlation)
            self.spin_structure_factors.append(spin_structure_factors)


    def observable(self):
        """
        Calculate the density of the current configuration of the auxiliary field X.
        """
        G_up, G_down = self.green_function()
        I = np.eye(self.N)

        # Single particle density matrices
        rho_up = I - G_up.T
        rho_down = I - G_down.T

        # Occupation numbers
        n_up = np.diag(rho_up)
        n_down = np.diag(rho_down)

        # Local spin polarization
        m = n_up - n_down

        # Spin correlation Matrix
        spin_correlation = 0.25 * (np.outer(m,m) - rho_up * rho_up.T - rho_down * rho_down.T + np.diag(n_up + n_down))

        # Density and double occupancy
        density = np.mean(n_up + n_down)
        double_occupancy = np.mean(n_up * n_down)

        # Antiferromagnetic structure factor
        S_AF = self.eta @ spin_correlation @ self.eta / self.N

        return density, double_occupancy, spin_correlation, S_AF


    def green_function(self):
        """
        Calculate the Green's function of the current configuration of the auxiliary field X.
        """
        return self.MC.G_up, self.MC.G_down

    def initialize_auxiliary_field(self):
        """
        Initialize the auxiliary field X with random values of +1 or -1.
        """
        X = np.random.choice([-1, 1], size=(self.M, self.N))
        return X
