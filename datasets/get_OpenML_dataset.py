""" I have been testing AutoML on 22 datasets from OpenML that are
    a subset of those listed in Balaji and Allen (2018)
    https://arxiv.org/pdf/1808.06492.pdf.
    These datasets require no preprocessing, e.g., to remove unsused string
    information.

    This module allows the user to download the dataset from OpenML
    by the number I have (somewhat arbitrarily given it), or to 
    download all such datasets.

    Usage:


    """

import argparse
import yaml

dataset_dict = yaml.load('dataset_info.yaml')

parser = argparse.ArgumentParser(description='Download OpenML dataset')

