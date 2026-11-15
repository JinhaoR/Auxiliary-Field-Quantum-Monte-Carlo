# Auxiliary Field Quantum Monte Carlo

## Model

In this project we will study the two dimensional repulsive Hubbard model with AFQMC and use it to reproduce the emergent antiferromagnetic correlations observed in ultra cold atom experiments.

The Model Hamiltonian is given by 

$$
H = -t \sum_{\braket{ij}, \sigma} (c_{i\sigma}^\dagger c_{j \sigma}^ + \text{h.c.}) + U \sum_i \left( n_{i\uparrow} - \frac{1}{2} \right) \left( n_{i \downarrow} - \frac{1}{2} \right).
$$

We assume that the lattice system is at half filling $\braket{n_{i \uparrow} + n_{i \downarrow}} = 1$. Working with a 2D square lattice let $N_s = L \times L$.

## Observable

The observable that we are interested are the double occupancy 

$$
D = \frac{1}{N_s} \sum_i \braket{n_{i \uparrow} n_{i \downarrow}} 
$$

and correlated spin.

## Implementation

To implement the algorithm, we will use $X$ as our auxiliery field, and $M$ imaginary time slices. Therefore X will have the shape $M \times N_s$.

We are going to calculate the partition function using the determinant identity for quadratic fermions 

$$
Z = C \sum_X W[X], \quad W[X] = \prod_\sigma \det(I + B_M^\sigma B_{M-1}^\sigma \cdots B_1^\sigma)
$$ 

where we have that 

$$
B_l^\sigma = e^{-\Delta \tau k / 2} e^{\sigma \lambda D_l} e^{-\Delta \tau k / 2} = T V_l^\sigma T
$$ 

where $\cosh \lambda = \exp (\Delta \tau U / 2) $ and $\Delta \tau = \beta / M$ is our imaginary time slice. The two matrices $k$ and $D_l$ are the single particle hopping matrix as well as the single particle diagonal matrix at imaginary time slice $l$ that we get after using HS transformation on our repulsive term.
