""" Class to obtain random functions used both for obtaining seed hyperparater points and
    possibly for benchmark data """

import numpy as np

def unit_vector(dim, embedded_dim):
    assert embedded_dim >= dim
    vec = np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    zeros = np.zeros(embedded_dim - dim)
    return np.hstack((vec, zeros))

class benchmark_function(object):

        def __init__(
                self, n_features, n_samples,
                random_mag, linear_mag, sin_kx_mag, kx_2_mag, kx_3_mag
                lin_dim, sin_kx_dim, kx_2_dim, kx_3_dim,
                lin_sign, sin_kx_sign, kx_2_sign, kx_3_sign,
                sin_kx_freq 
                )

            self.n_features = n_features
            self.n_samples = n_samples

            self.random_mag = random_mag
            self.linear_mag = linear_mag
            self.sin_kx_mag = sin_kx_mag
            self.kx_2_mag = kx_2_mag
            self.kx_3_mag = kx_3_mag

            self.lin_dim = lin_dim
            self.sin_kx_dim = sin_kx_dim
            self.kx_2_dim = kx_2_dim
            self.kx_3_dim = kx_3_dim

            self.lin_sign = lin_sign
            self.sin_kx_sign = sin_kx_sign
            self.kx_2_sign = kx_2_sign
            self.kx_3_sign = kx_3_sign

            self.sin_kx_freq = sin_kx_freq

            self.linear_unit_vector = unit_vector(lin_dim, n_features)
            self.structure_unit_vector = unit_vector(structure_dim, n_features)