""" Main .py for running autoML_plus """
from __future__ import print_function

import argparse
import numpy as np
import pickle
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import sys

import config

LOWER_RMSE_THRESHOLD, UPPER_RMSE_THRESHOLD = 0.1, 0.4
DEFAULT_POPULATION = 5

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-file_name')
parser.add_argument('--trainings', '-trainings')
parser.add_argument('--quick_stop', '-quick_stop')
parser.add_argument('--seed', '-seed')
parser.add_argument('--test_size', '-test_size')
parser.add_argument('--validation_size', '-validation_size')
parser.add_argument('--target_column', '-target_column')
parser.add_argument('--verbosity', '-verbosity')
parser.add_argument('--model', '-model')
args = parser.parse_args()

if args.file_name in [None, 'boston']:
    print('Using built-in Boston Housing Data.')
    print('Ignoring potential -target_column argument.')
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
        raise RuntimeError(
                          "Failed to read data file '" + 
                          input_file_name +
                          "' as CSV.  Aborting."
                          )

# default values
trainings = 100
quick_stop = 'NONE'
seed_value = 42
test_size = 0.25
validation_size = 0.1
target_column = -1
verbosity = 0

if args.trainings:
    trainings = int(args.trainings)
    if trainings not in [20, 100]:
        print(
             'Running with unvalidated option of', trainings, 
             'model trainings.'
             )

if args.quick_stop:
    quick_stop = args.quick_stop.upper()

if quick_stop not in ['NONE', 'AGRESSIVE', 'MODERATE']:
    raise ValueError('Unrecognized option for quick_stop')

if args.seed:
    seed_value = int(args.seed)

if args.test_size:
    test_size = float(args.test_size)
    if test_size < 0 or test_size >= 1.:
        raise ValueError('test_size must be > 0 and < 1.')

if args.validation_size:
    validation_size = float(args.validation_size)
    if validation_size < 0 or validation_size >= 1.:
        raise ValueError('validation_size must be > 0 and < 1.')

if test_size + validation_size >= 1.:
    raise ValueError('test_size + validation_size must be < 1.')

if args.target_column:
    target_column = int(args.target_column)

if args.verbosity:
    verbosity = int(args.verbosity)

if args.model:
    if args.model.upper() in ['DNN', 'LINEAR']:
        model = args.model.upper()
    else:
        raise ValueError('Unrecognized option for model')
else:
    model = None

# choosing target and other columns in general

if input_file_name == 'boston':
    pass
else:
    try:
        if target_column in [-1, data_shape[1] - 1]:
            data, target = file_data[:, :-1], file_data[:, -1:]
        elif target_column in [0, -data_shape[1]]:
            data, target = file_data[:, 1:], file_data[:, :1]
        else:
            target = file_data[:, target_column]
            data = np.hstack((
                             file_data[:, :target_column],
                             file_data[:, target_column + 1:]
                            ))
    except:
        raise ValueError(
                        'Cannot choose target_column ' + 
                        str(target_column)
                        )

    target = np.ravel(target)

train_size = 1. - (test_size + validation_size)
(
x_train, x_test,
y_train, y_test
) = train_test_split(
                    data, target, train_size=train_size,
                    test_size=test_size, random_state=seed_value
                    )

val_size = validation_size/train_size
train_size = 1. - val_size

(
x_train, x_val,
y_train, y_val
) = train_test_split(
                    x_train, y_train, train_size=train_size,
                    test_size=val_size, random_state=seed_value
                    )

population, generations = 1, 0
run_param = {
            'population_size': population,
            'verbosity': verbosity,
            'generations': generations,
            'random_state': seed_value
            }

if model == 'DNN':
    config_dict = config.NN_config_dictionary(*data.shape)
    run_param['config_dict'] = config_dict
elif model == 'LINEAR':
    config_dict = config.model_config_dict('linear')
    run_param['config_dict'] = config_dict

# If quick_stop options are selected, do 1 model training
if quick_stop != 'NONE':
    run_AutoML = True
    tpot = TPOTRegressor(**run_param)
    tpot.fit(x_train, y_train)
    y_predict = tpot.predict(x_val)
    rmse = np.sqrt(np.mean((y_predict - y_val)**2))
    mean_abs = np.mean(np.abs(y_val))
    rmse_scaled = rmse / mean_abs
    stop_lower = (rmse_scaled < LOWER_RMSE_THRESHOLD)
    stop_upper = (rmse_scaled < UPPER_RMSE_THRESHOLD)
    if stop_lower or (stop_upper and quick_stop == 'AGRESSIVE'):
        print('Quick Stop Criterion Realized.  Stopping after 1 training.')
        run_AutoML = False

if quick_stop == 'NONE' or run_AutoML:
    print('Running AutoML')
    run_param['population_size'] = DEFAULT_POPULATION
    generations = int(np.ceil(trainings / DEFAULT_POPULATION))
    run_param['generations'] = generations
    tpot = TPOTRegressor(**run_param)
    tpot.fit(x_train, y_train)
    print('Finished running AutoML.\n')

y_predict = tpot.predict(x_test)
rmse = np.sqrt(np.mean((y_predict - y_test)**2))
mean_abs = np.mean(np.abs(y_test))
rmse_scaled = rmse / mean_abs
print('RMSE on test data =', rmse)
print('Scaled RMSE on test data =', rmse_scaled, '\n')
tpot.export('output.py')
print('Optimal pipeline in output.py.')


