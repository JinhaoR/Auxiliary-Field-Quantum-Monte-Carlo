# Auxiliary Field Quantum Monte Carlo

In this project we will study the two dimensional repulsive Hubbard model with AFQMC and use it to reproduce the emergent antiferromagnetic correlations observed in ultra cold atom experiments.

The Model Hamiltonian is given by $$H = -t \sum_{\braket{ij}, \sigma} (c_{i\sigma}^\dagger c_{j \sigma}^\dagger + \text{h.c.}) + U \sum_i n_{i\uparrow} n_{i \downarrow}.$$
We assume that the lattice system is at half filling ${n_{i \sigma}} = 1$ and $U >> t$. Working with a 2D square lattice let $N_s = L \times L$.

The observable that we are interested are the double occupancy $$D = \frac{1}{N_s} \sum_i \braket{n_{i \uparrow} n_{i \downarrow}} $$ and correlated spin.