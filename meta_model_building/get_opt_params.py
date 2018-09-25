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
    if sys.argv[1] == 'boston':
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


# Set generations
meta_generations = 2

# Set population


seed = {
    'split_seed': seed_value,
    'tpot_seed': seed_value
    }

split_param = {
    'train_size': 0.75,
    'test_size': 0.25
    }

run_param = {
    'population_size': 10,
    'verbosity': 1,
    'generations': 1,
    'random_state': seed['tpot_seed'],
    'warm_start': True
    }

housing = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    data, target, train_size=split_param['train_size'],
    test_size=split_param['test_size'], random_state=seed['split_seed']
    )

tpot = TPOTRegressor(**run_param)
best_scores = []

start_time = time.time()
time_int = int(round(start_time))
output_name = input_file_name.split(".")[0] + '-' + str(time_int)
output_python = output_name + '.py'
output_pickle = output_name + '.pickle'

for i_gen in range(meta_generations):
    tpot.fit(x_train, y_train)
    score = tpot.score(x_test, y_test)
    best_scores.append(score)
    tpot.export(output_name + '-' + str(i_gen) + '.py')

finish_time = time.time()


# Prepare metadata dictionary for pickling
pickle_dict = tpot.evaluated_individuals_
pickle_dict.update(run_param)
pickle_dict.update(split_param)
pickle_dict.update(seed)
pickle_dict['meta_generations'] = meta_generations
pickle_dict['duration'] = finish_time - start_time
pickle_dict['output_pickle'] = output_pickle

with open(output_pickle, 'wb') as pickle_file:
    pickle.dump(pickle_dict, pickle_file)
