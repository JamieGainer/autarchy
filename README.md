# autarchy: User-Friendly AutoML with a Model Hierarchy (Insight Data Science Project)

## Use Case

The goal of this package is to provide [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning) 
for structured data in a useful way for data science professionals.
We focus on structured data (regression in the current version), due to its importance for business analytics.

In selecting an ML model, one often has to choose between complicated models, which can fit the data well but may overfit,
and simple models, which will not fit the data as well, but are less likely to overfit the data and which may be more 
human-understandable.
Also, AutoML, the choosing of the ML model, its architecture, and other hyperparameters automatically, is generally 
time-consuming, and time is often limited for users.

To solve these two issues simultaneously we present autarchy, a tool which takes a list of ML models, generally 
in increasing order of complexity, and finds good hyperparameters for each.  The user has the option of stopping
individual models after a certain amount of time and/or of stopping the entire process after a certain length of time
to ensure that one obtains the best possible answer for the user given the available time.

## Technical Description

autarchy is built on [TPOT](http://epistasislab.github.io/tpot/), a package for AutoML that uses 
[genetic programming](https://en.wikipedia.org/wiki/Genetic_programming)
for hyperparameter and model selection.


## Repo Format:
- **src** : Source code for production
- **tests** : Source code for tests
- **configs** : Configuration files for AutoML on particular models (configs/models) as well as for overall running 
(in configs parent directory)
- **data** : Sample data for tests and for for further user-defined testing or demo-ing.
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

## Requisites
- TPOT (not sure about version yet)
- TPOT requires sklearn, np (maybe others)


## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint 
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```

## To Do

# 


## Potential Future Directions/ Features

# Take categorical data