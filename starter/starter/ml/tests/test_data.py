""" Tests for starter/data.py """

# NOTE: this works because we included __init__.py files in both ml/ and ml/tests/
import os

import pandas as pd
import numpy as np
import pytest
import starter.ml.data as data
from sklearn.preprocessing import OneHotEncoder

@pytest.fixture
def data_path():
    """ Ensures that we can run pytest from everywhere
    in the terminal. 
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    return f"{file_dir}/test_data.csv"

@pytest.fixture
def test_df(data_path):
    return data.load_data(data_path)


def test_load_data(test_df):
    """Very simple test for the load_data function."""
    assert test_df.size > 0


def test_get_categorical_features(test_df):
    actual_cat_features_set = set(
        [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
    )
    cat_features_set = set(data.get_categorical_features(test_df))
    assert actual_cat_features_set == cat_features_set


def test_categorical_slices_dict():
    test_dict = {"a": [1, 2, 3, np.nan], "b": ["c", "e", "f", np.nan], "c": [0, np.nan, 0, 1]}
    test_df = pd.DataFrame(test_dict)
    slices_dict = data.categorical_slices_dict(test_df, ["a", "b", "c"])
    assert np.array_equal(slices_dict["a"], np.array([1, 2, 3]))
    assert np.array_equal(slices_dict["b"], np.array(["c", "e", "f"]))
    assert np.array_equal(slices_dict["c"], np.array([0, 1]))


def test_process_data_slice(test_df):
    """ Check (the length) of data slices of test_df. """
    categorical_features = data.get_categorical_features(test_df)
    first_cat_feature = categorical_features[0]
    values_dict = test_df[first_cat_feature].value_counts().to_dict()
    
    # this step is needed to obtain the encoders
    X, y, encoder, lb = data.process_data(test_df, categorical_features, "income", training=True)

    # check all (lengths of) data slices for the first_cat_feature
    for value, value_count in values_dict.items():
        data_slice, y_slice = data.process_data_slice(test_df, categorical_features, first_cat_feature, value, encoder, lb)
        #print(f"value = {value}: count = {value_count}")
        assert value_count == len(data_slice)
    
