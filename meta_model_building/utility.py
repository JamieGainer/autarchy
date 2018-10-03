""" This module contains methods called in the main script for obtaining data:
get_opt_params.py"""

import copy
import numpy as np
from tpot.config.regressor import regressor_config_dict
from tpot.config.regressor_light import regressor_config_dict_light

# model lists and dicts
implemented_model_list = [
    'regression', 'regression_light', 'knn',
    'decision_tree', 'linear', 'SVR', 'DNN'
    ]
linear_model_list = [
    'sklearn.linear_model.ElasticNetCV',
    'sklearn.linear_model.LassoLarsCV',
    'sklearn.linear_model.RidgeCV'
    ]
model_abbreviation_dict = {
    'knn': 'sklearn.neighbors.KNeighborsRegressor',
    'decision_tree': 'sklearn.tree.DecisionTreeRegressor',
    'SVR': 'sklearn.svm.LinearSVR'
    }
light_model_list = [
    'sklearn.linear_model.ElasticNetCV',
    'sklearn.tree.DecisionTreeRegressor',
    'sklearn.neighbors.KNeighborsRegressor',
    'sklearn.linear_model.LassoLarsCV',
    'sklearn.svm.LinearSVR',
    'sklearn.linear_model.RidgeCV'
    ]

# preprocessor lists and dicts
implemented_preprocessor_list = ['max_abs_scaler', 'min_max_scaler', 'PCA']
light_preprocessor_list = [
    'sklearn.preprocessing.Binarizer',
    'sklearn.cluster.FeatureAgglomeration',
    'sklearn.preprocessing.MaxAbsScaler',
    'sklearn.preprocessing.MinMaxScaler',
    'sklearn.preprocessing.Normalizer',
    'sklearn.kernel_approximation.Nystroem',
    'sklearn.decomposition.PCA',
    'sklearn.kernel_approximation.RBFSampler',
    'sklearn.preprocessing.RobustScaler',
    'sklearn.preprocessing.StandardScaler',
    'tpot.builtins.ZeroCount'
    ]
preprocessor_abbreviation_dict = {
    'max_abs_scaler': 'sklearn.preprocessing.MaxAbsScaler',
    'min_max_scaler': 'sklearn.preprocessing.MinMaxScaler',
    'PCA': 'sklearn.decomposition.PCA'
}


def model_config_dict(model_space):
    """Given string model_space in implemented_model_list, return the
    corresponding dictionary in a format appropriate to the TPOT class"""
    assert model_space in implemented_model_list
    if model_space == 'regression':
        return_dict = copy.deepcopy(regressor_config_dict)
    else:
        return_dict = copy.deepcopy(regressor_config_dict_light)
        if model_space == 'regression_light':
            pass
        elif model_space == 'linear':
            for model in light_model_list:
                if model not in linear_model_list:
                    return_dict.pop(model)
        else:
            for model in light_model_list:
                if model != model_abbreviation_dict[model_space]:
                    return_dict.pop(model)
    return return_dict


def restrict_preprocessor(preprocessor, config_dict):
    assert preprocessor in implemented_preprocessor_list
    for prep in light_preprocessor_list:
        if prep != preprocessor_abbreviation_dict[preprocessor]:
            config_dict.pop(prep)

def NN_config_dictionary(N_samples, N_features):
    n_layers = 5
    heuristic_first_layer = int(round(N_samples/(2. * N_features)))
    first_hidden = max(100, heuristic_first_layer)
    print(first_hidden)
    layers = [int(round(first_hidden**(i/n_layers))) for i in range(n_layers,0,-1)]
    return {
           'sklearn.neural_network.MLPRegressor': {
           'hidden_layer_sizes': [layers],
           'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.],
           'solver': ['lbfgs', 'sgd', 'adam']},

    # Preprocesssors
          'sklearn.preprocessing.Binarizer': {
          'threshold': np.arange(0.0, 1.01, 0.05)},

          'sklearn.cluster.FeatureAgglomeration': {
          'linkage': ['ward', 'complete', 'average'],
          'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']},

          'sklearn.preprocessing.MaxAbsScaler': {},

          'sklearn.preprocessing.MinMaxScaler': {},

          'sklearn.preprocessing.Normalizer': {
          'norm': ['l1', 'l2', 'max'] },

          'sklearn.kernel_approximation.Nystroem': {
          'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 
                     'poly', 'linear', 'additive_chi2', 'sigmoid'],
          'gamma': np.arange(0.0, 1.01, 0.05),
          'n_components': range(1, 11)},

          'sklearn.decomposition.PCA': {
          'svd_solver': ['randomized'],
          'iterated_power': range(1, 11)},

          'sklearn.kernel_approximation.RBFSampler': {
          'gamma': np.arange(0.0, 1.01, 0.05)},

          'sklearn.preprocessing.RobustScaler': {},

          'sklearn.preprocessing.StandardScaler': {},

          'tpot.builtins.ZeroCount': {},

    # Selectors
          'sklearn.feature_selection.SelectFwe': {
          'alpha': np.arange(0, 0.05, 0.001),
          'score_func': {'sklearn.feature_selection.f_regression': None}},

          'sklearn.feature_selection.SelectPercentile': {
          'percentile': range(1, 100),
          'score_func': {
          'sklearn.feature_selection.f_regression': None}},

          'sklearn.feature_selection.VarianceThreshold': {
          'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]}
    }
