# -*- encoding = utf-8 -*-

import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'
FILE_ROOT = '../resources/'
HOUSING_FILE_PATH = FILE_ROOT + HOUSING_PATH


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_FILE_PATH):
    print(housing_url)
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


if __name__ == '__main__':
    fetch_housing_data()
