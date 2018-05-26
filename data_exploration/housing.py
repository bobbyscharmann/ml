"""
May 23, 2018

    Download and untar a datafile. Liberally taken from
    Hands on Machine Learning by Aurelien Geron page 44.

Scharmann
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer

# Install development version of scikit-learn by using
# to use CategoricalEncoder 
# pip install git+git://github.com/scikit-learn/scikit-learn.git
from sklearn.preprocessing import OneHotEncoder, CategoricalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from six.moves import urllib
import tarfile

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

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

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

if __name__ == "__main__":
    fetch_housing_data()
    housing = load_housing_data()
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    print(housing.describe())
    # housing.hist(bins=50, figsize=(20,15))
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population", 
                 figsize=(10,7), c="median_house_value", 
                 cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    # plt.show()    
    train_set, test_set = split_train_test(housing, 0.2)
    print(len(train_set), "train +", len(test_set), "test")
    corr_matrix = housing.corr()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()


    attributes = ["median_house_value", "median_income", "total_rooms"]
    # scatter_matrix(housing[attributes], figsize=(20,7))
    # plt.show()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    
    # Use an SimpleImputer to set missing values to the median
#    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
 #  imputer.fit(housing_num)
 #  print(imputer.statistics_)
 #  X = imputer.transform(housing_num)
 #   housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    
    # Prepare one-hot encoding for categorical values
    housing_cat = housing["ocean_proximity"]
    housing_cat_encoded, housing_categories = housing_cat.factorize()

    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

    num_attributes = list(housing_num)
    cat_attributes = ["ocean_proximity"]
 
    num_pipeline = Pipeline([
                            ('selector', DataFrameSelector(num_attributes)),
                            ('imputer', SimpleImputer(strategy="median")),
                            ('std_scaler', StandardScaler()),
                            ])

    cat_pipeline = Pipeline([
                            ('selector', DataFrameSelector(cat_attributes)),
                            ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
                            ])
    full_pipeline = FeatureUnion(transformer_list=[
                        ("num_pipeline", num_pipeline),
                        ("cat_pipeline", cat_pipeline),
                        ])

    housing_prepared = full_pipeline.fit_transform(housing)
    
    m = LinearRegression()
    m.fit(housing_prepared, housing_labels)
    housing_predictions = m.predict(housing_prepared)
    m_mse = mean_squared_error(housing_labels, housing_predictions)
    m_rmse = np.sqrt(m_mse)
    print(m_rmse)



    print("Finished\n")












