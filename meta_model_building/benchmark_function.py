""" Class to obtain random functions used both for obtaining seed hyperparater points and
    possibly for benchmark data """

import numpy as np

def unit_vector(dim, embedded_dim):
    if dim > embedded_dim:
        dim = embedded_dim
    vec = np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    zeros = np.zeros(embedded_dim - dim)
    return np.hstack((vec, zeros))

def linear(k, x):
    return x.dot(k)

def sin_kx_phase(k, x, freq, phase):
    return np.sin(freq * linear(k,x) + phase)

def kx_2(k, x):
    return linear(k, x)**2

def kx_3(k, x):
    return linear(k, x)**3


class benchmark_function(object):

        def __init__(
                    self, n_features,
                    random_mag, linear_mag, sin_kx_mag, kx_2_mag, 
                    kx_3_mag, lin_dim, sin_kx_dim, kx_2_dim, kx_3_dim,
                    lin_sign, sin_kx_sign, kx_2_sign, kx_3_sign,
                    sin_kx_freq, sin_kx_phase, seed=42
                    ):

            self.n_features = n_features

            self.random_mag = random_mag
            self.sin_kx_freq = sin_kx_freq
            self.sin_kx_phase = sin_kx_phase
            self.seed = seed

            self.linear_dict = {
                                'name': 'linear',
                                'mag': linear_mag,
                                'dim': linear_dim,
                                'sign': linear_sig
                               }

            self.sin_kx_dict = {
                                'name': 'sin_kx',
                                'mag': sin_kx_mag,
                                'dim': sin_kx_dim,
                                'sign': sin_kx_sig
                               }

            self.kx_2_dict = {
                                'name': 'kx_2',
                                'mag': kx_2_mag,
                                'dim': kx_2_dim,
                                'sign': kx_2_sig
                               }

            self.kx_3_dict = {
                                'name': 'kx_3',
                                'mag': kx_3_mag,
                                'dim': kx_3_dim,
                                'sign': kx_3_sig
                               }

            self.param_dicts = (
                               self.linear_dict, self.sin_kx_dict,
                               self.kx_2_dict, self.kx_3_dict
                               )

            self.mags = (d['mag'] for d in self.param_dicts)

            self.normalization = self.random_mag + sum(self.mags)

            self.unit_vector_dict = {
                                    d['name']: self.unit_vector(d['dim'], self.n_features)
                                    for d in self.param_dicts
                                    }

            def full_linear(x):
                return (self.linear_dict['mag'] * self.linear_dict['sign'] * 
                    linear(self.unit_vector_dict['linear'], x))

            def full_sin_kx(x):
                return (self.sin_kx_dict['mag'] * self.sin_kx_dict['sign'] * 
                    linear(self.unit_vector_dict['sin_kx'], x,
                        self.sin_kx_freq, self.sin_kx_phase))

            def full_kx_2(x):
                return (self.kx_2_dict['mag'] * self.kx_2_dict['sign'] * 
                    kx_2(self.unit_vector_dict['kx_2'], x))

            def full_kx_3(x):
                return (self.kx_3_dict['mag'] * self.kx_3_dict['sign'] * 
                    kx_3(self.unit_vector_dict['kx_3'], x))

            def full_random(x):
                np.random.seed(self.seed)
                return self.random_mag * np.random.rand(x.shape[0])

            def function_values(x):
                return (
                        full_linear(x),
                        full_sin_kx(x),
                        full_kx_2(x),
                        full_kx_3(x),
                        full_random(x)
                        )

            def function(x):
                return sum(function_values(x))/self.normalization












