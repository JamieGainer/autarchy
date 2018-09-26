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

if len(sys.argv) > 1:
    input_file_name = sys.argv[1]
    if input_file_name == 'boston':
        housing = load_boston()
        data, target = housing.data, housing.target
    else:
        try:
            file_data = np.genfromtxt(input_file_name, delimiter=',')
            data, target = file_data[:, :-1], file_data[:, -1:]
            target = np.ravel(target)
        except:
            print(
                "Failed to read data file", input_file_name,
                "as CSV.  Aborting."
                )
            quit()
else:
    input_file_name = 'boston'
    housing = load_boston()
    data, target = housing.data, housing.target

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

epochs = 10
if '-epochs' in sys.argv:
    epoch_position = sys.argv.index('-epochs')
    try:
        epochs = int(sys.argv[epoch_position + 1])
    except:
        print(
            'Epochs cannot be set to specified value. ',
            'Aborting.'
            )
        quit()

generations_per_epoch = 5
if '-generations_per_epoch' in sys.argv:
    gen_position = sys.argv.index('-generations_per_epoch')
    try:
        generations_per_epoch = int(sys.argv[gen_position + 1])
    except:
        print(
            'Generations per epoch cannot be set to specified value. ',
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
        assert verbosity in [0,1,2,3]
    except:
        print(
              'Verbosity cannot be set to the specified value. ',
              'Aborting.'
             )

print('Parameters:')
print('Input file name:', input_file_name)
print('Hyperparameter space:', model_space)
print('Preprocessor:', preprocessor)
print('Epochs:', epochs)
print('Generations per epoch:', generations_per_epoch)
print('Population:', population)
print('Seed value:', seed_value)
print('Verbosity:', verbosity)
print(config_dict.keys())
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
    'generations': generations_per_epoch,
    'random_state': seed['tpot_seed'],
    'config_dict': config_dict,
    'warm_start': True
    }

x_train, x_test, y_train, y_test = train_test_split(
    data, target, train_size=split_param['train_size'],
    test_size=split_param['test_size'], random_state=seed['split_seed']
    )

tpot = TPOTRegressor(**run_param)
train_scores
test_scores = []

start_time = time.time()
time_int = int(round(start_time))
output_name = input_file_name.split(".")[0] + '-' + str(time_int)
output_python = output_name + '.py'
output_pickle = output_name + '.pickle'

for i_gen in range(epochs):
    print(i_gen)
    tpot.fit(x_train, y_train)
    train_scores.append(tpot.score(x_train, y_train))
    test_scores.append(tpot.score(x_test, y_test))
    best_pipelines.append(tpot._optimized_pipeline)
    tpot.export(output_name + '-' + str(i_gen) + '.py')
print('train:', train_scores)
print('test:', test_scores)


finish_time = time.time()


# Prepare metadata dictionary for pickling
pickle_dict = {}
pickle_dict['evaluated_individuals'] = tpot.evaluated_individuals_
pickle_dict.update(run_param)
pickle_dict.update(split_param)
pickle_dict.update(seed)
pickle_dict['test_scores'] = test_scores
pickle_dict['train_scores'] = train_scores

pickle_dict['model_space'] = model_space
pickle_dict['preprocessor'] = preprocessor
pickle_dict['epochs'] = epochs
pickle_dict['generations_per_epoch'] = generations_per_epoch
pickle_dict['population'] = population
pickle_dict['seed_value'] = seed_value
pickle_dict['conifg_dict'] = config_dict

pickle_dict['duration'] = finish_time - start_time
pickle_dict['output_pickle'] = output_pickle

with open(output_pickle, 'wb') as pickle_file:
    pickle.dump(pickle_dict, pickle_file)
