import autarchy
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

housing = load_boston()
train_size = autarchy.config_dict['train-test']['train_size']
test_size = autarchy.config_dict['train-test']['test_size']

X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25)

generations, population_size, verbosity = (autarchy.config_dict['genetic'][key]
                                           for key in ['generations', 'population_size', 
                                                       'verbosity'])
tpot = autarchy.tpot.TPOTRegressor(generations=generations, 
                                   population_size=population_size, 
                                   verbosity=verbosity)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
