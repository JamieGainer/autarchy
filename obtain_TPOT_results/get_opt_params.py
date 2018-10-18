""" Script used to 

Invoked:

python get_opt_params.py -file_name file_name [ optional arguments ]

The optional arguments are
-generations
-population
-seed
-target_column
-model_space
-preprocessor
-verbosity

"""

import argparse
import numpy as np
import pickle
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import sys
import time
import config

model_list = config.implemented_model_list
preprocessor_list = config.implemented_preprocessor_list

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-file_name')
parser.add_argument('--generations', '-generations')
parser.add_argument('--population', '-population')
parser.add_argument('--seed', '-seed')
parser.add_argument('--target_column', '-target_column')
parser.add_argument('--model_space', '-model_space')
parser.add_argument('--preprocessor', '-preprocessor')
parser.add_argument('--verbosity', '-verbosity')
args = parser.parse_args()

if args.file_name == None:
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
seed_value, target_column = 42, -1
generations, population = 5, 50
verbosity = 0
model_space = 'regression'

if args.seed:
    seed_value = int(args.seed)

if args.target_column:
    target_column = int(args.target_column)

if args.generations:
    generations = int(args.generations)

if args.population:
    population = int(args.population)

if args.verbosity:
    verbosity = int(args.verbosity)

if args.model_space:
    model_space = args.model_space
    if model_space not in model_list:
        raise ValueError('Could not read model_space.  Aborting.')

if model_space.upper() == 'DNN':
    config_dict = config.NN_config_dictionary(*data_shape)
else:
    config_dict = config.model_config_dict(model_space)

preprocessor = args.preprocessor
if preprocessor:
    if preprocessor not in preprocessor_list:
        raise ValueError(
                        "Cannot choose preprocessing method '" +  
                        str(preprocessor) + '\b.  Aborting.'
                        )
    config.restrict_preprocessor(preprocessor, config_dict)

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

print('Parameters:')
print('Input file name:', input_file_name)
print('Hyperparameter space:', model_space)
print('Pre-chosen Preprocessor:', preprocessor)
print('Generations:', generations)
print('Population:', population)
print('Seed value:', seed_value)
print('Verbosity:', verbosity)
print()

seed = {
    'split_seed': seed_value,
    'tpot_seed': seed_value
    }

split_param = {
    'train_size': 0.75,
    'test_size': 0.25
    }

run_param = {
    'population_size': population,
    'verbosity': verbosity,
    'generations': generations,
    'random_state': seed['tpot_seed'],
    'config_dict': config_dict,
    'warm_start': True
    }

x_train, x_test, y_train, y_test = train_test_split(
    data, target, train_size=split_param['train_size'],
    test_size=split_param['test_size'], random_state=seed['split_seed']
    )

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=13./14.,
    test_size=1./14., random_state=seed['split_seed']
    )

split_param['train_size'] = 0.65
split_param['validation_size'] = 0.10

tpot = TPOTRegressor(**run_param)
train_scores = []
val_scores = []
test_scores = []
cv_scores = []
best_pipelines = []


start_time = time.time()
time_int = int(round(start_time))
output_name = input_file_name.split(".")[0] + '-' + str(time_int)
output_python = output_name + '.py'
output_pickle = output_name + '.pickle'


tpot.fit(x_train, y_train)
train_scores.append(tpot.score(x_train, y_train))
test_scores.append(tpot.score(x_test, y_test))
val_scores.append(tpot.score(x_val, y_val))
best_pipelines.append(tpot._optimized_pipeline)
cv_scores.append(max([x.fitness.values[1] for x in tpot._pop]))
tpot.export(output_name + '-0.py')

train_scores = np.array(train_scores)
val_scores = np.array(val_scores)
test_scores = np.array(test_scores)
cv_scores = np.array(cv_scores)
mean_train_target = np.mean(y_train)
mean_test_target = np.mean(y_test)
mean_data = np.mean(target)
median_train_target = np.median(y_train)
median_test_target = np.median(y_test)
median_data = np.median(target)

finish_time = time.time()


# Prepare metadata dictionary for pickling
pickle_dict = {}
pickle_dict.update(run_param)
pickle_dict.update(split_param)
pickle_dict.update(seed)

pickle_dict['test_scores'] = test_scores
pickle_dict['val_scores'] = val_scores
pickle_dict['train_scores'] = train_scores
pickle_dict['cv_scores'] = cv_scores
pickle_dict['mean_train_target'] = mean_train_target
pickle_dict['mean_test_target'] = mean_test_target
pickle_dict['mean_data'] = mean_data
pickle_dict['median_train_target'] = median_train_target
pickle_dict['median_test_target'] = median_test_target
pickle_dict['median_data'] = median_data

pickle_dict['model_space'] = model_space
pickle_dict['preprocessor'] = preprocessor
pickle_dict['generations'] = generations
pickle_dict['population'] = population
pickle_dict['seed_value'] = seed_value
pickle_dict['conifg_dict'] = config_dict

pickle_dict['duration'] = finish_time - start_time
pickle_dict['output_pickle'] = output_pickle
pickle_dict['input_file_name'] = input_file_name

with open(output_pickle, 'wb') as pickle_file:
    pickle.dump(pickle_dict, pickle_file)
