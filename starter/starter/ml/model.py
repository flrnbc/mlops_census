from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

CURRENT_DIR = Path(__file__).parent.resolve() # TODO: refactor

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


def get_encoders():
    """ 
    Load encoders e.g. for model inference.  
    NOTE: They only exist after first training.
    """
    try:
        encoders_dir = CURRENT_DIR.parents[2]/"encoders"
        encoder = joblib.load(encoders_dir/"one_hot_encoder.sav")
        lb = joblib.load(encoders_dir/"label_binarizer.sav")
    except Exception as exc:
        logging.error("Encoders could not be loaded.")
        raise exc
    return encoder, lb


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


def inference(model, X):
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
    except ValueError:
        print("Not a correct input for the model")
        return np.array([[]])
    return pred
