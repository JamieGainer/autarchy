""" I have been testing AutoML on 22 datasets from OpenML that are
    a subset of those listed in Balaji and Allen (2018)
    https://arxiv.org/pdf/1808.06492.pdf.
    These datasets require no preprocessing, e.g., to remove unsused string
    information.

    This module allows the user to download the dataset from OpenML
    by the number I have (somewhat arbitrarily given it), or to 
    download all such datasets.

    Usage:

    get_OpenML_dataset.py n

    downloads datset labelled by the integer n

    get_OpenML_dataset.py all

    download all datasets


    """

def download_from_URL(url, file_name):
    import requests

    with open(file_name, "wb") as csv_file:
        print('Downloading', file_name, '...')
        response = requests.get(url)
        csv_file.write(response.content)


def download_dataset(n, dataset_dict):
    import os

    dataset_entry = dataset_dict[n]
    file_name = dataset_entry['file_name']
    url = dataset_entry['url']

    if file_name in os.listdir('.'):
        print(file_name, 'already downloaded.')
    else:
        download_from_URL(url, file_name)


import sys
import yaml

try:
    command = sys.argv[1]
except IndexError:
    print('Please specify dataset number or "all" to download all datasets.')

with open('dataset_info.yaml', 'r') as yaml_file:
    dataset_dict = yaml.load(yaml_file)

if command.upper() == 'ALL':
    for key in dataset_dict:
        download_dataset(key, dataset_dict)
else:
    try:
        command = int(command)
    except ValueError:
        pass
    if command in dataset_dict:
        download_dataset(command, dataset_dict)
    else:
        print(command, 'not found')





