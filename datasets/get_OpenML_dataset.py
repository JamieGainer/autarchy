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

def download_from_URL(url):
    pass


def download_dataset(n, dataset_dict):
    import os
    dataset_entry = dataset_dict[n]
    file_name = dataset_entry['file_name']
    if file_name in os.listdir('.'):
        print(file_name, 'already downloaded.')
    else:
        download_from_URL(dataset_dict[n]['url'])

import sys
import yaml

print(len(sys.argv))

with open('dataset_info.yaml', 'r') as yaml_file:
    dataset_dict = yaml.load(yaml_file)



