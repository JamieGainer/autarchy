""" Main .py for running autarchy """
from __future__ import print_function

import numpy as np
from sklearn import model_selection
import sys
from tpot import TPOTRegressor

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
	'random_state_for_split': 42,
	'generations': 5,
	'population_size': 20,
	'verbosity': 2,
	'random_state_tpot': 42,
	'min_random_initial_points': 5
}

# divide into X_train, X_test, y_train, y_test

X, y = data[:, :-1], data[:, -1:] 
y = np.ravel(y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
	train_size=run_param['train_size'], test_size=run_param['test_size'],
	random_state = run_param['random_state_for_split'])

# load benchmark data

benchmark_dict = {}

# pick set of benchmarks based on number of features and number of samples

choose_benchmark_key = lambda x, y: None

benchmark_key = choose_benchmark_key(benchmark_dict.keys(), data.shape)

try:
	benchmarks = benchmark_dict[benchmark_key]
except KeyError:
	benchmarks = set([])


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
