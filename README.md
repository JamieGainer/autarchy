# autarchy: Fast AutoML for Structured Data

## Use Case

The goal of this package is to provide [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning) 
for structured data in a useful way for data science professionals.
We focus on structured data (regression in the current version), due to its importance for business analytics.

Slides discussing this project are available [here] (https://bit.ly/2O6biok).


## Technical Description and Basic Usage

The main script (autarchy) in the autarchy directory performs a quick hyperparmeter tuning run.  It is built on [TPOT](http://epistasislab.github.io/tpot/).  

The main command is 
```
python autarchy -file_name file_name -target_column target_column -quick_stop quick_stop -trainings trainings
```
If -file_name is omitted, the Boston Housing Dataset built into scikit-learn will be used instead of a csv file.  If file_name is specified, it should be the pathname of a CSV file.  The column


## Repo Format:
- **autarchy** : Package directory for autarchy.
- **obtain_TPOT_results** : Code for running TPOT to obtain data about how well AutoML performs.
- **datasets** : Code to download datasets/ info about benchmark datasets.
- **TPOT_results_archive** : Some data about running TPOT on the datasets.  Organized by AutoML run, then by dataset number.
- **tests**: Tests.
