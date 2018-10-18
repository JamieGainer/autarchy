""" Main .py for running autarchy """
from __future__ import print_function

import argparse
import numpy as np
import pickle
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import sys

LOWER_RMSE_THRESHOLD, UPPER_RMSE_THRESHOLD = 0.1, 0.4

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-file_name')
parser.add_argument('--trainings', '-trainings')
parser.add_argument('--quick_stop', '-quick_stop')
parser.add_argument('--seed', '-seed')
parser.add_argument('--test_size', '-test_size')
parser.add_argument('--feature_column', '-feature_column')
parser.add_argument('--verbosity', '-verbosity')
args = parser.parse_args()

if args.file_name in [None, 'boson']:
    print('Using built-in Boston Housing Data') # log
    input_file_name = 'boston'
    housing = load_boston()
    data, target = housing.data, housing.target
    data_shape = (506, 14)
else:
    input_file_name = args.file_name
    try:
        file_data = np.genfromtxt(input_file_name, delimiter=',', skip_header=1)
        data_shape = file_data.shape
    except:
        print(
            "Failed to read data file", input_file_name,
            "as CSV.  Aborting."
            ) # log
        quit()


# default values
trainings = 20
quick_stop = 'NONE'
seed_value = 42,
test_size = 0.25
feature_column = -1
verbosity = 0

if args.trainings:
    trainings = int(args.trainings)

if args.quick_stop:
    quick_stop = args.quick_stop.upper()

if quick_stop not in ['NONE', 'AGRESSIVE', 'MODERATE']:
    print('Unrecognized option for quick_stop') #log
    quit()

if args.seed:
    seed_value = int(args.seed)

if args.test_size:
    test_size = float(args.test_size)

if args.feature_column:
    feature_column = int(args.feature_column)

if args.verbosity:
    verbosity = int(args.verbosity)

# choosing target and other columns in general
if input_file_name == 'boston':
    pass
else:
    try:
        if feature_column in [-1, data_shape[1] - 1]:
            data, target = file_data[:, :-1], file_data[:, -1:]
        elif feature_column in [0, -data_shape[1]]:
            data, target = file_data[:, 1:], file_data[:, :1]
        else:
            target = file_data[:, feature_column]
            data = np.hstack((
                             file_data[:, :feature_column],
                             file_data[:, feature_column + 1:]
                            ))
    except:
        print('Cannot choose feature_column', feature_column)
        print('Aborting.')
        quit()
    target = np.ravel(target)


train_size = 1. - test_size
(
x_train, x_test,
y_train, y_test
) = train_test_split(
                    data,  target, train_size=train_size,
                    test_size, random_state=seed_value
                    )

val_size = 0.1/train_size
train_size -= 1. - val_size

(
x_train, x_val,
y_train, y_val
) = train_test_split(
                    x_train, y_train, train_size=train_size,
                    test_size=val_size, random_state=seed_value
                    )

population = 1, generations = 0
run_param = {
            'population_size': population,
            'verbosity': verbosity,
            'generations': generations,
            'random_state': seed_value
            }

# If quick_stop options are selected, do 1 model training
if quick_stop != 'NONE':
    tpot = TPOTRegressor(**run_param)
    tpot.fit(x_train, y_train)
    y_predict = tpot.predict(x_val)
    rmse = np.sqrt(np.mean((y_predict - y_val)**2))
    mean_abs = np.mean(np.abs(y_val))
    rmse_scaled = rmse / mean_abs
    if rmse_scaled < LOWER_RMSE_THRESHOLD




# Train models using benchmark points to get scores for these models

score = lambda x: None

benchmarks_with_scores = {benchmark: score(benchmark) for benchmark in benchmarks}

# Merge benchmark hyperparameter points into TPOT regressor object

merge_models_into_tpot = lambda x, y: None

merge_models_into_tpot(tpot, benchmarks_with_scores)

# Resume tpot running with new model set

continue_tpot = lambda x, generations = run_param['generations']: None

continue_tpot(tpot, generations = run_param['generations'])

# TPOT will print out best model to stdout
