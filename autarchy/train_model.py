""" Stuff

    """

import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras

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

print(data_shape)

