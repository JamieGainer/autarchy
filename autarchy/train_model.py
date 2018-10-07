""" Stuff

    """

import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras

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

param_dict = {
             'target_column': -1,
             'seed': 42
             }

for run_param in run_params:
    argument = '_' + run_param
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


