from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Optional: implement hyperparameter tuning.
def get_model(X_train: np.array, y_train: np.array, mode: str="train", save: bool=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    mode: str
        Either 'train' for training or "load" for loading a model
    Returns
    -------
    model
        Trained machine learning model.
    """
    modes = ["train", "load"]
    if mode not in modes:
        raise ValueError("Not a known mode.")
    models_dir = Path(__file__).resolve().parents[3]/"models"
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
