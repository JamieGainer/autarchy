# autarchy: User-Friendly AutoML with a Model Hierarchy

## Use Case

The goal of this package is to provide [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning) 
for structured data in a useful way for data science professionals.
We focus on structured data (regression in the current version), due to its importance for business analytics.


## Technical Description and Basic Usage
The main script (autarchy) in the autarchy directory performs a quick hyperparmeter tuning run.  It is built on [TPOT](http://epistasislab.github.io/tpot/).  

```
python autarchy -file_name file_name -quick_stop quick_stop -trainings trainings
```

```
python autarchy
```
runs TPOT for up to about 100 model trainings, but tries to stop quicker, based on data from hyperparameter tuning. 
```
python autarchy -full
```
Runs TPOT with 100 model trainings.  
```
python autarchy -random
```
Runs TPOT for a single model training for a randomly selected hyperparameter point.


## Repo Format:
- **autarchy** : Package directory for autarchy.
- **obtain_TPOT_results** : Code for running TPOT to obtain data about how well AutoML performs.
- **datasets** : Code to download datasets/ info about benchmark datasets.
- **TPOT_results_archive** : Some data about running TPOT on the datasets.  Organized by AutoML run, then by dataset number.
