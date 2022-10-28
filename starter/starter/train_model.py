# Script to train machine learning model.
from multiprocessing.sharedctypes import Value
import os
from typing import Dict, List
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import data, model
from ml.data import categorical_slices_dict
from ml.model import compute_model_metrics
import ml.slice_metrics as slm

# load data as dataframe
# so that we can execute the program from "everywhere in the terminal"
# TODO: better way, e.g. using working dir as env variable?
curr_file_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = curr_file_dir + "/../../data/census.csv"
df = data.load_data(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

# process train data
cat_features = data.get_categorical_features(train, exclude=["income"])
LABEL = "income"
X_train, y_train, encoder, lb = data.process_data(
    X=train, categorical_features=cat_features, label=LABEL, training=True
)
# save encoder and lb
encoders_dir = Path(__file__).resolve().parents[2]/"encoders"
joblib.dump(encoder, encoders_dir/"one_hot_encoder.sav")
joblib.dump(lb, encoders_dir/"label_binarizer.sav")

# proces test data 
X_test, y_test, encoder, lb = data.process_data(
    X=test,
    categorical_features=cat_features,
    label=LABEL,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model and save
trained_model = model.get_model(X_train, y_train, mode="train", save=True)

# performance on train data
pred_test = model.inference(trained_model, X_test)
# print(pred_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, pred_test)

# performance on slices
perf_dict = slm.slices_performance(trained_model, test_df=test, save=True)  # note defaults
print(perf_dict)
