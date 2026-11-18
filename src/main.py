from runtime import RuntimeDeterminant, RuntimeGreenFunction
from model import HubbardModel
import numpy as np
from matplotlib import pyplot as plt

L=8
t=1.0
U=0.0
beta=4.0
M=int(beta * 10)
HubbardModel = HubbardModel(L=L, t=t, U=U)

thermalization = 1000
measure = 1000

Runtime = RuntimeGreenFunction(HubbardModel, M=M, beta=beta)

Runtime.run(thermalization, measure)

print(f"U = {U}, t = {t}, L = {L}, beta = {beta}, M = {M}")

#Let's print the observables of the system after the simulation with mean and error with np 
densitys = Runtime.densitys
mean_density = np.mean(densitys)
print(f"Density: {mean_density}")

double_occupancies = Runtime.double_occupancies
mean_double_occupancy = np.mean(double_occupancies)
print(f"Double Occupancy: {mean_double_occupancy}")

spin_correlations = Runtime.spin_correlations
mean_spin_correlation = np.mean(spin_correlations, axis=0)
print(f"Spin Correlation: {mean_spin_correlation}")

S_AFs = Runtime.spin_structure_factors
mean_S_AF = np.mean(S_AFs)
print(f"Antiferromagnetic structural factor: {mean_S_AF}")


def spatial_correlations(C, L):
    result = np.zeros((L, L))

    for dy in range(L):
        for dx in range(L):

            for y in range(L):
                for x in range(L):

                    i = x + L * y

                    jx = (x + dx) % L
                    jy = (y + dy) % L

                    j = jx + L * jy

                    result[dy, dx] += C[i, j]

    return result / (L * L)

C_spatial = spatial_correlations(mean_spin_correlation, L)

plt.imshow(C_spatial, cmap="RdBu_r")
plt.colorbar(label=r"$C^{zz}(\Delta x,\Delta y)$")
plt.xlabel(r"$\Delta x$")
plt.ylabel(r"$\Delta y$")
plt.show()

