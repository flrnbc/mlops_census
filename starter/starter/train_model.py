# Script to train machine learning model.
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import data, model
from ml.data import categorical_slices_dict
from ml.model import compute_model_metrics

# load data as dataframe
# so that we can execute the program from "everywhere in the terminal"
# TODO: better way, e.g. using working dir as env variable?
curr_file_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = curr_file_dir + "/../../data/census.csv"
df = data.load_data(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

# train data
cat_features = data.get_categorical_features(train, exclude=["income"])
LABEL = "income"
X_train, y_train, encoder, lb = data.process_data(
    X=train, categorical_features=cat_features, label=LABEL, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = data.process_data(
    X=test,
    categorical_features=cat_features,
    label=LABEL,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model
trained_model = model.train_model(X_train, y_train)

# performance on train data
pred_test = model.inference(trained_model, X_test)
# print(pred_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, pred_test)

# performance on all categorical data slices
cat_slices_dict = categorical_slices_dict(test, cat_features=cat_features)


def slices_performance(
    trained_model,
    test_df: pd.DataFrame,
    save: bool = False,
    output_dir: str = ".",
    cat_slices_dict: Dict[str, List] = cat_slices_dict,
) -> dict:
    """Compute performance on categorical data slices."""
    perf_dict = {}
    performance_scores = ["precision", "recall", "fbeta"]
    # TODO: check that we do not mess up the orderings along the way
    for slice, slice_values in cat_slices_dict.items():
        slice_dict = {score: [] for score in performance_scores}
        slice_indices = []  # for row names
        for value in slice_values:
            # row to save the performances for the data slice with this fixed value
            slice_indices.append(f"{slice}={value}")  # used as row labels
            # TODO: again: improve this without processing test_df over and over
            array_slice, y_slice = data.process_data_slice(
                test_df,
                categorical_features=cat_features,
                slice_col=slice,
                slice_value=value,
                encoder=encoder,
                lb=lb,
            )
            performance = list(
                compute_model_metrics(
                    y_slice, model.inference(trained_model, array_slice)
                )
            )
            # precision, recall, fbeta = compute_model_metrics(y_slice, model.inference(trained_model, array_slice))
            for idx, score in enumerate(performance_scores):
                slice_dict[score].append(performance[idx])
            slice_df = pd.DataFrame(slice_dict, index=slice_indices)
            perf_dict[slice] = slice_df
        if save:
            output_file = output_dir + "/" + slice + ".csv"  # TODO: fix path
            slice_df.to_csv(path_or_buf=output_file, index=slice_indices)
    return perf_dict


perf_dict = slices_performance(trained_model, test_df=test, save=True)  # note defaults
print(perf_dict)
