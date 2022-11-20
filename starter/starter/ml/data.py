import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

CURRENT_DIR = Path(__file__).parent.resolve() # TODO: refactor

def get_categorical_features(
    df: pd.DataFrame, exclude: List[str] = ["income"]
) -> List[str]:
    """This function retrieves the categorical features of a dataframe.

    More precisely, it collects all column names of the dataframe whose
    corresponding column has dtype == object with first entry a string
    (`StringDtype` might be better...)

    Inputs
    ------
    df: pd.DataFrame

    exclude: List[str] == ["income"]
        List of names which will be excluded from categorical features even if
        it fulfills the above criteria.
        NOTE: the default is adjusted to this project

    Returns
    -------
    List[str]
    """
    cat_features = []
    non_empty_cols = [
        col for col in df.columns if len(df[col]) > 0 and col not in exclude
    ]
    for col in non_empty_cols:
        if df[col].dtype == object:  # TODO: and type(df[col][0]) == str:
            cat_features.append(col)
    return cat_features


def categorical_slices_dict(
    df: pd.DataFrame, cat_features: list, remove_nan: bool = True
) -> Dict[str, List]:
    """Returns a dict with cat features-(unique) values as key-value pairs.
    """
    # TODO: cat_features = get_categorical_features() gave a bug...
    slices_dict = {}
    for feature in cat_features:
        if remove_nan:
            slices_dict[feature] = df[feature].dropna().unique()
        else:
            slices_dict[feature] = df[feature].unique()
    return slices_dict


# load data
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    # get absolute path
    p = p.resolve()
    print(p)
    if not p.exists():
        raise FileExistsError
    return pd.read_csv(path)


# data processing and slices
def process_data(
    X: pd.DataFrame,
    categorical_features: List[str] = [],
    label: str = None,
    training: bool = True,
    encoder=None,
    lb=None,
) -> tuple:
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    print(X_categorical)
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(
            X_categorical
        )  # TODO: what if encoder = None?
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def process_data_slice(
    df: pd.DataFrame,
    categorical_features: List[str],
    slice_col: str,
    slice_value,
    encoder,
    lb,
) -> np.array:
    """Create and process a data slice with an encoder."""
    try:
        df_slice = df.loc[df[slice_col] == slice_value]
        assert len(df_slice) > 0
    except KeyError:
        raise KeyError(f"{slice_col} is not a column name.")
    except AssertionError:
        raise AssertionError(
            f"Get empty data slice for column {slice_col} and value {slice_value}."
        )

    array_slice, y_slice, encoder, lb = process_data(
        df_slice,
        categorical_features,
        label="income",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return array_slice, y_slice


def get_encoders():
    """ 
    Load encoders e.g. for model inference.  
    NOTE: They only exist after first training.
    """
    try:
        encoders_dir = CURRENT_DIR.parents[2]/"encoders"
        encoder = joblib.load(encoders_dir/"one_hot_encoder.sav")
        lb = joblib.load(encoders_dir/"label_binarizer.sav")
    except FileNotFoundError:
        logging.error("Encoders could not be loaded.")
        raise FileNotFoundError
    return encoder, lb


# def process_for_inference(data):
#     """ 
#     Helper function to process a dataframe before inference.
#     """
#     # if we already have a numpy array, return
#     if isinstance(data, np.ndarray):
#         return data
#     # else transform via encoders
#     cat_features = get_categorical_features(data, exclude=[]) # nothing to exclude because of inference
#     encoder, lb = get_encoders()
#     X_np, y_np, encoder, lb = process_data(data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
#     return X_np

    #print("Prediction: " + lb.inverse_transform(pred))