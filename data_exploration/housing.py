"""
May 23, 2018

    Download and untar a datafilei. Liberally taken from
    Hands on Machine Learning by Aurelien Geron page 44.

Scharmann
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.plotting import scatter_matrix
from six.moves import urllib
import tarfile

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Download hosing data and extract it
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Convert the housing CSV to a Pandas dataframe
def load_housing_data(housing_path=HOUSING_PATH, housing_csv="housing.csv"):
    csv_path = os.path.join(housing_path, housing_csv)
    return pd.read_csv(csv_path)

# Split data into train and test sets
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":
    fetch_housing_data()
    housing = load_housing_data()

    print(housing.describe())
    housing.hist(bins=50, figsize=(20,15))
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population", 
                 figsize=(10,7), c="median_house_value", 
                 cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    # plt.show()    
    train_set, test_set = split_train_test(housing, 0.5)
    print(len(train_set), "train +", len(test_set), "test")
    corr_matrix = housing.corr()

    attributes = ["median_house_value", "median_income", "total_rooms"]
    scatter_matrix(housing[attributes], figsize=(20,7))
    plt.show()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    print("Finished\n")
