""" Main .py for running autarchy """
from __future__ import print_function

import numpy as np
import sys
import tpot

assert len(sys.argv) > 1
input_file_name = sys.argv[1]

# currently limited to csv: dependent variable is last column
try:
	data = np.genfromtxt(input_file_name, delimiter=',')
except:
	print("Failed to read data file", input_file_name)
	quit()

# run dictionary with default parameters that (later) can be overwritten with user input
run_param = {
	'train_size': 0.75,
	'test_size': 0.25,
	'random_state_for_split': 42
}

# divide into X_train, X_test, y_train, y_test

X, y = data[:, :-1], data[:, -1:] 

X_train, X_test, y_train, y_test = train_test_split(X, y,
	train_size=run_param['train_size'], test_size=run_param['test_size'],
	random_state = run_param['random_state_for_split'])

# load benchmark data

benchmark_dict = {}

# pick set of benchmarks based on number of features and number of samples

choose_benchmark_key = lambda x: None

benchmark_key = choose_benchmark_key(benchmark_dict.keys(), data.shape)

try:
	benchmarks = benchmark_dict[benchmark_key]
except KeyError:
	benchmarks = set([])

