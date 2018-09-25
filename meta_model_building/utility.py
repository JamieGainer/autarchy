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