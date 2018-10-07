""" Stuff

    """

import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def baseline_general(n_features, n_hidden):
    model = Sequential()
    model.add(
             Dense(
                   n_hidden, 
                   input_dim=n_features, 
                   kernel_initializer='normal',
                   activation='relu'
                  )
             )
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

run_params = ['target_column', 'seed']

try:
    input_file_name = sys.argv[1]
except:
    print('Need csv file name as first input.')
    quit()

try:
    file_data = np.genfromtxt(input_file_name, delimiter=',', skip_header=1)
except:
    print('Unable to read data file:', input_file_name)
    quit()

data_shape = file_data.shape
n_features = data_shape[1] - 1
n_samples = data_shape[0]

param_dict = {
             'target_column': -1,
             'seed': 42
             }

for run_param in run_params:
    argument = '-' + run_param
    if argument in sys.argv:
        try:
            index = sys.argv.index(argument)
            value = int(sys.argv[index + 1])
            param_dict[run_param] = value
        except:
            print(
                 'Value of', argument, 'must be integer provided',
                 'after', argument, '\b.')
            quit()

for run_param in run_params:
    print(run_param, '\b:', param_dict[run_param])

try:
    y = file_data[:, param_dict['target_column']]
except IndexError:
    print(param_dict['target_column'], 'is not a valid column.')
    quit()

mask = (file_data == file_data)
mask[:, param_dict['target_column']] = False

x = file_data[mask].reshape((n_samples, n_features))

hidden_layer_n = 10

np.random.seed(param_dict['seed'])
baseline_model = baseline_general(n_features, 13)
estimator = KerasRegressor(
                          build_fn=baseline_model,
                          epochs=100,
                          batch_size=5,
                          verbose=0
                          )




