""" Tests for starter/model.py """
import numpy as np
from pytest import approx
import sklearn

import starter.ml.model as model
import starter.ml.data as data

def test_train_model():
    """ Simple smoke test for training a model for binary classification """
    X = np.random.randn(100, 30)
    y = np.random.randint(2, size=100) # binary
    mod = model.get_model(X, y)
    assert type(mod) == sklearn.linear_model.LogisticRegression # TODO: store type of model e.g. in params?


def test_compute_model_metrics():
    """ Testing compute_model_metrics with simple examples """
    y1 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y1_pred = np.array([1, 1, 0, 0, 0, 0, 0, 1])
    """ fp = 2, tp = 1, fn = 3,  
        -> precision_score = tp/(tp+fp) = 1/3
           recall_score = tp/(tp+fn) = 1/4
           fbeta = (1+beta^2)*(precsion*recall)/(beta^2*precision+recall)
                 = 2/7 (for beta=1)
    """
    # use approx to avoid issues with precision
    assert (1/3, 1/4, approx(2/7, abs=1e-10)) == model.compute_model_metrics(y1, y1_pred) 


def test_inference(test_df):
    """ Test inference """
    categorical_features = data.get_categorical_features(test_df)
    X, y, encoder, lb = data.process_data(test_df, categorical_features, "income", training=True)

    mod = model.get_model(X, y)
    y_pred = model.inference(mod, X)

    assert type(model.inference(mod, X)) == np.ndarray
    assert model.inference(mod, X).shape == y.shape