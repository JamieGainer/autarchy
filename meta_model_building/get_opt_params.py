import os
import pickle
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import time

run_param = {
    'random_state_train': 42,
    'random_state_tpot': 42,
    'export_python_code_filename': 'tpot_boston_pipeline.py',
    'pickle_file_name': 'boston_pipeline_evaluated_models.pickle',
    'train_size': 0.75,
    'test_size': 0.25,
    'generations': 5,
    'population_size': 20,
    'verbosity': 2,
    'time': time.time()
}

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=run_param['train_size'], 
                                                    test_size=run_param['test_size'],
                                                    random_state = run_param['random_state_train'])

tpot = TPOTRegressor(generations=run_param['generations'], population_size=run_param['population_size'],
                    verbosity=run_param['verbosity'], random_state = run_param['random_state_tpot'])

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export(run_param['export_python_code_filename'])

# Prepare metadata dictionary for pickling
pickle_dict = tpot.evaluated_individuals_
pickle_dict.update(run_param)


with open(run_param['pickle_file_name'], 'w') as pickle_file:
    pickle.dump(pickle_dict, pickle_file)