from runtime import Runtime
from model import HubbardModel
import numpy as np


HubbardModel = HubbardModel(L=8, t=1.0, U=4.0)

thermalization = 10
measure = 10

Runtime = Runtime(HubbardModel, M=10, beta=1.0)

Runtime.run(thermalization, measure)

#Let's print the observables of the system after the simulation with mean and error with np 
densitys = Runtime.densitys
mean_density = np.mean(densitys)
error_density = np.std(densitys) / (len(densitys) ** 0.5)
print(f"Density: {mean_density} ± {error_density}")

double_occupancies = Runtime.double_occupancies
mean_double_occupancy = np.mean(double_occupancies)
error_double_occupancy = np.std(double_occupancies) / (len(double_occupancies) ** 0.5)
print(f"Double Occupancy: {mean_double_occupancy} ± {error_double_occupancy}")


