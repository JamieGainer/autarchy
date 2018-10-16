import numpy as np
import os
import pickle
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import sys
import time
import utility

model_list = utility.implemented_model_list
preprocessor_list = utility.implemented_preprocessor_list


if len(sys.argv) == 1 or sys.argv[1] == 'boston':
    input_file_name = 'boston'
else:
    input_file_name = sys.argv[1]
if input_file_name == 'boston':
    housing = load_boston()
    data, target = housing.data, housing.target
    data_shape = (506, 14)
else:
    try:
        file_data = np.genfromtxt(input_file_name, delimiter=',', skip_header=1)
        data_shape = file_data.shape
    except:
        print(
            "Failed to read data file", input_file_name,
            "as CSV.  Aborting."
            )
        quit()

if '-seed' in sys.argv:
    seed_position = sys.argv.index('-seed')
    try:
        seed_value = int(sys.argv[seed_position + 1])
        print('Random seeds set to', seed_value)
    except:
        seed_value = 42
        print('Random seeds set to 42')
else:
    seed_value = 42
    print('Random seeds set to 42')

feature_column = -1
if '-feature_column' in sys.argv:
    fc_position = sys.argv.index('-feature_column')
    try:
        feature_column = int(sys.argv[fc_position + 1])
        if input_file_name == 'boston':
            if feature_column not in [-1, 14]:
                print(
                     'Cannot set different feature column',
                     'for Boston housing data.'
                     )
                raise exception
        print('feature column set to', feature_column)
    except:
        print('Misunderstood feature column.  Aborting.')
        quit()

print('feature_column = ', feature_column)

if '-model_space' in sys.argv:
    ms_position = sys.argv.index('-model_space')
    try:
        model_space = sys.argv[ms_position + 1]
        if model_space not in model_list:
            print('Could not read model_space.  Aborting')
            quit()
    except:
        print('Could not read model_space.  Aborting')
        quit()
else:
    model_space = 'regression'
print('Model space set to', model_space)
if model_space.upper() == 'DNN':
    config_dict = utility.NN_config_dictionary(*data_shape)
else:
    config_dict = utility.model_config_dict(model_space)

preprocessor = None
if '-preprocessor' in sys.argv:
    prep_position = sys.argv.index('-preprocessor')
    try:
        preprocessor = sys.argv[prep_position + 1]
        if preprocessor not in preprocessor_list:
            print(
                'Cannot choose preprocessing method', preprocessor,
                '\b.  Aborting.'
                )
            quit()
    except:
        print('Could not choose preprocessing method.  Aborting')
        quit()
if preprocessor:
    utility.restrict_preprocessor(preprocessor, config_dict)

generations = 5
if '-generations' in sys.argv:
    gen_position = sys.argv.index('-generations')
    try:
        generations = int(sys.argv[gen_position + 1])
    except:
        print(
            'Generations cannot be set to specified value. ',
            'Aborting.')
        quit()

population = 50
if '-population' in sys.argv:
    pop_position = sys.argv.index('-population')
    try:
        population = int(sys.argv[pop_position + 1])
    except:
        print(
            'Population cannot be set to the specified value. ',
            'Aborting.'
            )
        quit()

verbosity = 0
if '-verbosity' in sys.argv:
    verb_position = sys.argv.index('-verbosity')
    try:
        verbosity = int(sys.argv[verb_position + 1])
        assert verbosity in [0, 1, 2, 3]
    except:
        print(
              'Verbosity cannot be set to the specified value. ',
              'Aborting.'
             )

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

print('Parameters:')
print('Input file name:', input_file_name)
print('Hyperparameter space:', model_space)
print('Preprocessor:', preprocessor)
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
