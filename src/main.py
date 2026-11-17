from runtime import Runtime
from model import HubbardModel
import numpy as np


HubbardModel = HubbardModel(L=8, t=1.0, U=4)

thermalization = 10
measure = 100

beta = 1.0
M = 10

Runtime = Runtime(HubbardModel, M, beta)

Runtime.run(thermalization, measure)

#Let's print the density of the system after the simulation with mean and error with np 
densitys = Runtime.densitys
mean_density = np.mean(densitys)
error_density = np.std(densitys) / (len(densitys) ** 0.5)
print(f"Density: {mean_density} ± {error_density}")
