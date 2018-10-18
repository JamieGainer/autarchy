""" Main .py for running autarchy """
from __future__ import print_function

import argparse
import numpy as np
import pickle
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import sys

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

# Initialize regressor that runs once
tpot = TPOTRegressor(**run_param)



# Start tpot.  We will replace some of the initial hyperparmeter points with benchmark points

population_size = max(run_param['population_size'], len(benchmarks) + run_param['min_random_initial_points'])
tpot = TPOTRegressor(generations=0, population_size=population_size, 
					 verbosity=run_param['verbosity'], random_state = run_param['random_state_tpot'])

tpot.fit(X_train, y_train)

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
