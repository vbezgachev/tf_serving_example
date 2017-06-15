import os
from os.path import isfile
from urllib.request import urlretrieve

from scipy.io import loadmat
from dl_progress import DLProgress


data_dir = 'data/'

def download_train_and_test_data():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not isfile(data_dir + "train_32x32.mat"):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                data_dir + 'train_32x32.mat',
                pbar.hook)

    if not isfile(data_dir + "test_32x32.mat"):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                data_dir + 'test_32x32.mat',
                pbar.hook)


def load_data_sets():
    if not os.path.exists(data_dir):
        raise Exception("Data directory doesn't exist!")

    train_set = loadmat(data_dir + 'train_32x32.mat')
    test_set = loadmat(data_dir + 'test_32x32.mat')
    return train_set, test_set


def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min())/(255 - x.min()))

    # scale to feature_range
    min_val, max_val = feature_range
    x = x * (max_val - min_val) + min_val
    return x
