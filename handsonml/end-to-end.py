# -*- encoding = utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from handsonml.downloadfile import HOUSING_FILE_PATH
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


def load_housing_data(housing_path, file_name):
    csv_path = os.path.join(housing_path, file_name)
    return pd.read_csv(csv_path)


def split_test_set(data_set, test_ratio):
    shuffled_indices = np.random.permutation(len(data_set))
    test_set_size = int(len(data_set) * test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data_set.iloc[train_indices], data_set.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    """

    :param data:
    :param test_ratio:
    :param id_column:
    :param hash:
    :return: train_set, test_set
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def stratified_split_test_by_income(data, test_ration):
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)
    print(data['income_cat'].max())
    print(data['income_cat'].value_counts() / len(data))

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ration, random_state=42)
    for train_index, test_index in split.split(data, data['income_cat']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    for item in (strat_train_set, strat_test_set):
        item.drop(['income_cat'], axis=1, inplace=True)

    strat_train_set.to_csv(path_or_buf="../resources/datasets/housing/housing_train.csv")
    strat_test_set.to_csv(path_or_buf="../resources/datasets/housing/housing_test.csv")

    return strat_train_set, strat_test_set


def deal_with_missing_value(data):
    # seperate the predictors and the labels
    housing = data.drop("median_house_value", axis=1)
    housing_label = data["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    # imputer cannot deal with text attribute
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    print(imputer.statistics_)
    print(housing_num.median().values)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    return housing_tr


def handling_text_category_attributes_1hot(data):
    # encoder = LabelEncoder()
    housing_cat = data["ocean_proximity"]
    # housing_cat_encoded = encoder.fit_transform(housing_cat)
    # print(encoder.classes_)
    # print(housing_cat_encoded)
    # print(type(housing_cat.array))
    print(housing_cat.to_numpy().reshape(-1, 1))

    one_hot_encoder = OneHotEncoder(categories='auto')
    housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat.to_numpy().reshape(-1, 1))
    print(housing_cat_1hot.toarray())


def handling_text_category_attributes_label_binaries(data):
    housing_cat = data["ocean_proximity"]
    encoder = LabelBinarizer()
    housing_cat_lb = encoder.fit_transform(housing_cat)
    print(housing_cat_lb)


class CombineAttributesAdder(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, add_bedrooms_per_room=True):  # no *args or ##kargs, add_bedrooms_per_room is a hyperparameter
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedroom_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    """
    此处借鉴https://stackoverflow.com/questions/46162855/
    fit-transform-takes-2-positional-arguments-but-3-were-given-with-labelbinarize
    ，但在后续运行模型过程中发现对'ocean_proximity'编码有问题，不同的数据集有不同的编码长度，后发现是之前这个类的代码每次都初始化
    LabelBinarizer，导致每次重新fit，做如下修改后解决了问题，结论：编码器在一次运行中要保证只能fit一次
    """
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
        self.enc = LabelBinarizer(sparse_output=self.sparse_output)

    def fit(self, X, y=None):
        return self.enc.fit(X)

    def transform(self, X, y=None):
        return self.enc.transform(X)


def create_num_pipeline():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombineAttributesAdder()),
        ('str_scaler', StandardScaler())
    ])

    return num_pipeline


def create_full_pipeline(data):
    num_attribs = list(data)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_addr', CombineAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

    label_binarizer = CustomLabelBinarizer()
    # label_binarizer.fit()
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', label_binarizer)
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
    ])

    return full_pipeline


def train_with_linear_regression(data_prepared, labels, some_data, pl):
    """

    :param data_prepared:
    :param labels:
    :param some_data:
    :param pl:
    :return:
    """
    lin_reg = LinearRegression()
    lin_reg.fit(data_prepared, labels)

    # print("somedata \t ", some_data.shape)
    # some_labels = labels.iloc[:5]
    # some_data_prepared = pl.transform(some_data)
    # print("some data perepared\t", some_data_prepared.shape)
    #
    # print("Predictions:\t", lin_reg.predict(some_data_prepared))
    # print("Labels:\t\t", list(some_labels))

    housing_predictions = lin_reg.predict(data_prepared)
    lin_mse = mean_squared_error(labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)

    lin_scores = cross_val_score(lin_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)


def train_with_decision_tree(data_prepared, labels):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, labels)
    housing_prediction = tree_reg.predict(data_prepared)
    tree_mse = mean_squared_error(labels, housing_prediction)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)

    scores = cross_val_score(tree_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)


def train_with_random_forest(data_prepared, labels, X_test_prepared, y_test):
    forest_reg = RandomForestRegressor()
    # forest_reg.fit(data_prepared, labels)
    # forest_prediction = forest_reg.predict(data_prepared)
    # forest_mse = mean_squared_error(labels, forest_prediction)
    # forest_rmse = np.sqrt(forest_mse)
    # print(forest_rmse)
    #
    # scores = cross_val_score(forest_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10)
    # rmse_scores = np.sqrt(-scores)
    # display_scores(rmse_scores)

    # fine tune model
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_label)

    print("grid best params:", grid_search.best_params_)

    # evalute on test set
    final_model = grid_search.best_estimator_
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)


def display_scores(scores):
    print("Scores:", scores)
    print("mean:", scores.mean())
    print("Standard deviation:", scores.std())


if __name__ == '__main__':
    train_df = load_housing_data(HOUSING_FILE_PATH, 'housing_train.csv')
    test_df = load_housing_data(HOUSING_FILE_PATH, 'housing_test.csv')
    housing_cat = train_df["ocean_proximity"].copy()
    # seperate the predictors and the labels
    housing = train_df.drop("median_house_value", axis=1)
    print(list(housing))
    print(housing.shape)
    housing_label = train_df["median_house_value"].copy()
    # imputer cannot deal with text attribute
    housing_num = housing.drop("ocean_proximity", axis=1)
    # print(list(housing_num))

    X_test = test_df.drop('median_house_value', axis=1)
    y_test = test_df['median_house_value'].copy()

    # print(type(train_df.values))
    # train_df_num = deal_with_missing_value(train_df)
    # handling_text_category_attributes_1hot(train_df)
    # handling_text_category_attributes_label_binaries(train_df.iloc[:5])

    # attr_adder = CombineAttributesAdder(add_bedrooms_per_room=False)
    # housing_extra_attribs = attr_adder.transform(train_df.values)
    # print(housing_extra_attribs.shape, housing_extra_attribs[0])

    # num_pipeline = create_num_pipeline()
    # housing_num_tr = num_pipeline.fit_transform(housing_num)
    # print(housing_num_tr)

    full_pipeline = create_full_pipeline(housing_num)
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared.shape)

    X_test_prepared = full_pipeline.transform(X_test)

    # housing_prepared2 = full_pipeline.transform(housing.iloc[5:100])
    # print(housing_prepared2.shape)
    # train_with_linear_regression(housing_prepared, housing_label, housing[:5], full_pipeline)
    # print('\ntree')
    # train_with_decision_tree(housing_prepared, housing_label)
    print('\nforest')
    train_with_random_forest(housing_prepared, housing_label, X_test_prepared, y_test)





