""" This module contains methods called in the main script for obtaining data:
get_opt_params.py"""

import copy
from tpot.config.regressor import regressor_config_dict
from tpot.config.regressor_light import regressor_config_dict_light

# model lists and dicts
implemented_model_list = [
    'regression', 'regression_light', 'knn',
    'decision_tree', 'linear', 'SVR'
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

#preprocessor lists and dicts (to come)

