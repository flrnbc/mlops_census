from pathlib import Path

import joblib
import pandas as pd

from ml import data, model
from ml.data import categorical_slices_dict, get_categorical_features
from ml.model import compute_model_metrics

CURRENT_DIR = Path(__file__).parent

def slice_value_performance(
    trained_model, test_df: pd.DataFrame, slice: str, value: str, cat_features: list
) -> list:
    """Compute performance on a (categorical) data slice"""
    # TODO: again: improve this function without processing test_df over and over

    # load encoders (requires training before) from parent dir
    try:
        parent_dir = CURRENT_DIR # TODO: fix path for saving models
        encoder = joblib.load(parent_dir/"one_hot_encoder.sav")
        lb = joblib.load(parent_dir/"label_binarizer.sav")
    except:
        raise FileNotFoundError("Please check if encoders were correctly trained.")

    array_slice, y_slice = data.process_data_slice(
        test_df,
        categorical_features=cat_features,
        slice_col=slice,
        slice_value=value,
        encoder=encoder,
        lb=lb,
    )
    return compute_model_metrics(y_slice, model.inference(trained_model, array_slice))


def slice_performance(
    trained_model,
    test_df: pd.DataFrame,
    slice: str,
    save: bool = False,
    output_dir: str = "slices_metrics",
) -> pd.DataFrame:
    """ Compute performance for all values of a slice """
    cat_features = get_categorical_features(test_df)
    cat_slices_dict = categorical_slices_dict(test_df, cat_features)
    slice_values = cat_slices_dict[slice]
    # introduced to be more flexible if e.g. more scores are added
    performance_scores = [
        "precision",
        "recall",
        "fbeta",
    ]  # TODO: also store as params/envs?
    slice_dict = {score: [] for score in performance_scores}
    slice_indices = []  # for keeping track of row names
    for value in slice_values:
        # row to save the performances for the data slice with this fixed value
        slice_indices.append(f"{slice}={value}")  # used as row labels
        performance = slice_value_performance(
            trained_model, test_df, slice, value, cat_features
        )
        for idx, score in enumerate(performance_scores):
            slice_dict[score].append(performance[idx])
    slice_df = pd.DataFrame(slice_dict, index=slice_indices)
    if save:
        output_dir = CURRENT_DIR/output_dir
        file_name = slice + ".txt"
        output_file = output_dir/file_name
        slice_df.to_csv(path_or_buf=output_file, index=slice_indices)
    return slice_df


def slices_performance(
    trained_model,
    test_df: pd.DataFrame,
    save: bool = False,
    output_dir: str = "slices_metrics",
) -> dict:
    """Compute performance on categorical data slices."""
    cat_features = get_categorical_features(test_df)
    cat_slices_dict = categorical_slices_dict(test_df, cat_features)
    perf_dict = {}
    # TODO: check that we do not mess up the orderings along the way
    for slice in cat_slices_dict.keys():
        perf_dict[slice] = slice_performance(
            trained_model=trained_model,
            test_df=test_df,
            slice=slice,
            save=save,
            output_dir=output_dir
        )
    return perf_dict