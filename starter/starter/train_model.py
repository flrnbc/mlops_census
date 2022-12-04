# Script to train machine learning model.
from typing import Dict, List
from pathlib import Path

import joblib
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from starter.starter.ml import data, model
from starter.starter.ml.data import categorical_slices_dict
from starter.starter.ml.model import compute_model_metrics
import starter.starter.ml.slice_metrics as slm

# setup logging
logging.basicConfig(filename='train_model.log', level=logging.INFO, format='%(asctime)s %(message)s')


def train_model(label: str):
    """ Trains the model specified in ml/model.py and
    saves the encoders.
    """
    # load data as dataframe so that we can execute the program from "everywhere in the terminal"
    # TODO: better way, e.g. using working dir as env variable?
    curr_file_dir = Path(__file__).parent.resolve()
    data_path = curr_file_dir.parents[1]/"data/census.csv"
    df = data.load_data(data_path)

    # TODO: Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(df, test_size=0.20)
    # process train data
    cat_features = data.get_categorical_features(train, exclude=[label])
    logging.info(f"Categorical features: {cat_features}")
    X_train, y_train, encoder, lb = data.process_data(
        X=train, categorical_features=cat_features, label=label, training=True
    )

    # save encoder and lb
    encoders_dir = Path(__file__).resolve().parents[2]/"encoders"
    # TODO: add time stamp
    joblib.dump(encoder, encoders_dir/"one_hot_encoder.sav")
    logging.info(f"{type(encoder)} saved to {encoders_dir/'one_hot_encoder.sav'}")
    joblib.dump(lb, encoders_dir/"label_binarizer.sav")
    logging.info(f"{type(lb)} saved to {encoders_dir/'label_binarizer.sav'}")

    # proces test data 
    X_test, y_test, encoder, lb = data.process_data(
        X=test,
        categorical_features=cat_features,
        label=label,
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
    logging.info(f"Precision: {precision_test}, Recall: {recall_test}, fbeta: {fbeta_test}")
    # performance on slices
    perf_dict = slm.slices_performance(trained_model, test_df=test, save=True)  # note defaults
    #print(perf_dict)


if __name__ == "__main__":
    train_model(label="income") # TODO: read label from config/env vars?
