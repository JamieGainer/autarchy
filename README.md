# autoML_plus: Fast AutoML for Structured Data

## Use Case

The goal of this package is to provide [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning) 
for structured data in a useful way for data science professionals.
We focus on structured data (regression in the current version), due to its importance for business analytics.

Slides discussing this project are available [here](https://bit.ly/2O6biok).  I developed this project as an Insight AI Fellow in Fall 2018.


## Technical Description and Basic Usage

The main script (autoML_plus) in the autoML_plus directory performs a hyperparmeter tuning run.  It is built on [TPOT](http://epistasislab.github.io/tpot/) but has the following additional features

1. It recommends running for fewer model trainigs (20 or 100)
2. It has a quick stop option for when RMSE is already small after one training
3. It has a deep neural network option, which is useful on non-linear data

The main command is 
```
python autoML_plus.py [ options ]
```

The options are file_name, trainings, quick_stop, seed, test_size, target_column, verbosity, and model.  Each option can be specified using
```
-option value
```
or
```
--option value
```
after "python autoML_plus.py" on the command line.  All options are optional.

**file_name**
If file_name is specified, it should be the pathname of a CSV file.  If it is omitted, the Boston Housing Dataset built into scikit-learn will be used instead of a csv file.

**trainings**
The default value is 100.  Any integer value can be used, though the number of trainings implemented will be rounded up to the nearest 5.

**quick_stop**
The default value is "NONE".  To 

**seed**
The default value is 42.  Any integer value may be specified.

**test_size**
The default value is 0.25.  

**target_column**

The default value is -1 (the right most column), but other integer (positive or negative) values can be specified following the python indexing convention.

**verbosity**

Default value is 0 (quite).  This sets the extent to which TPOT sends progress messages to stdout.  Options are 1, 2, and 3.

**model**

Default is the full set of regression models in TPOT ().


## Repo Format:
- **autoML_plus** : Package directory for autarchy.
- **obtain_TPOT_results** : Code for running TPOT to obtain data about how well AutoML performs.
- **datasets** : Code to download datasets/ info about benchmark datasets.
- **TPOT_results_archive** : Some data about running TPOT on the datasets.  Organized by AutoML run, then by dataset number.
- **tests**: Tests.
