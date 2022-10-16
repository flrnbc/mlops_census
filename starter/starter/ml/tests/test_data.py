""" Tests for starter/data.py """

# NOTE: this works because we included __init__.py files in both ml/ and ml/tests/
import os
from pyexpat.errors import XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING

import numpy as np
import pandas as pd
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


def test_data_slice_from_array(test_df):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 2, 8],
            "b": [2, 3, 5, 1, 2, 1, 5],
            "level": ["med", "med", "high", "high", "low", "mid", np.nan],
        }
    )
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_categorical = encoder.fit_transform(np.array([df["level"].values]))
    print(df["level"].values)
    X, y, encoder, lb = data.process_data(df, categorical_features=["level"], label="b", training=True)

    # get data slice from X
    mapping_cols = data.get_mapping_cols(df, label="b")
    slice = "level"
    print(encoder.n_features_in_)
    slice_value = encoder.transform(np.array([["high"]]))[0]  # get one-hot-encoding for "high"
    data_slice = data.data_slice_from_array(X, slice, slice_value, mapping_cols)

    # test length
    assert len(data_slice) == 2
