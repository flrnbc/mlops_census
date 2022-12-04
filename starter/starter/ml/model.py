import logging
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from starter.starter.ml.data import (get_categorical_features, get_encoders,
                             process_data)

CURRENT_DIR = Path(__file__).parent.resolve() # TODO: refactor e.g. using envs, config file or ...

# Optional: implement hyperparameter tuning.
def get_model(X_train: Optional[np.array]=None, y_train: Optional[np.array]=None, mode: str="train", save: bool=False):
    """
    Trains a machine learning model and returns it.
    NOTE: the defaults for X_train, y_train enables us to just call get_model(mode="load").

    Inputs
    ------
    X_train : np.array=None
        Training data.
    y_train : np.array=None
        Labels.
    mode: str="train"
        Either 'train' for training or "load" for loading a model
    Returns
    -------
    model
        Trained machine learning model.
    """
    modes = ["train", "load"]
    if mode not in modes:
        raise ValueError("Not a known mode.")
    models_dir = CURRENT_DIR.parents[2]/"models"
    if mode == "train":
        lr = LogisticRegression(C=1.0)  # TODO: add as parameter
        lr.fit(X_train, y_train)
        logging.info(f"{type(lr)} trained")
        if save:
            joblib.dump(lr, models_dir/"trained_model.sav") # TODO: path as  parameter/env
            logging.info(f"Model saved to {models_dir/'trained_model.sav'}")
    elif mode == "load":
        lr = joblib.load(models_dir/"trained_model.sav")
    return lr


def compute_model_metrics(y: np.array, preds: np.array) -> tuple:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X: np.ndarray) -> np.ndarray:
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    try:
        pred = model.predict(X)
    except ValueError: # TODO: check for better exceptions
        logging.error("Model inference failed, returning empty np array.")
        return np.array([[]])
    return pred


def inference_pd(model, X: pd.DataFrame, decode: bool=False) -> np.ndarray:
    """ Run model inference with a dataframe, e.g. from a request.
    
    NOTE: This requires the encoders in data/encoders.

    Inputs
    ------
    model:
        Trained model
    X: pd.DataFrame
        DataFrame for inference
    decode: bool
        If true, use the encoder to transform from numerical to categorical predictions 

    Returns
    -------
    pred: np.ndarray
        Predictions
    """
    try: 
        encoder, lb = get_encoders()
        cat_features = get_categorical_features(X, exclude=[]) # nothing to exclude because of inference
        X, y, encoder, lb = process_data(X, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
        pred = inference(model, X)
        if decode:
            pred = lb.inverse_transform(pred)
    except Exception as exc: # TODO: better exceptions!
        logging.error("Inference with a DataFrame failed. Check encoders.")
        raise exc # TODO: return something else?        
    return pred
    

