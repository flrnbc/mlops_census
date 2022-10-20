# Script to train machine learning model.
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import data, model
from ml.data import categorical_slices_dict
from ml.model import compute_model_metrics

# load data as dataframe
# TODO: fix path
DATA_PATH = "../data/census.csv"
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
#print(pred_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(
    y_test, pred_test
)

# performance on all categorical data slices
cat_slices_dict = categorical_slices_dict(test, cat_features=cat_features)

def slices_performance(
    trained_model, test_df: pd.DataFrame, cat_slices_dict: Dict[str, List] = cat_slices_dict
) -> dict:
    """Compute performance on categorical data slices."""
    perf_dict = {}
    for slice, slice_values in cat_slices_dict.items():
        slice_dict = {}
        for value in slice_values:
            # TODO: again: improve this without processing test_df over and over
            array_slice, y_slice = data.process_data_slice(
                test_df,
                categorical_features=cat_features,
                slice_col=slice,
                slice_value=value,
                encoder=encoder,
                lb=lb,
            )
            precision, recall, fbeta = compute_model_metrics(y_slice, model.inference(trained_model, array_slice))
            slice_dict[value] = precision, recall, fbeta
        perf_dict[slice] = slice_dict
    return perf_dict

perf_dict = slices_performance(trained_model, test_df=test) # note default
print(perf_dict)