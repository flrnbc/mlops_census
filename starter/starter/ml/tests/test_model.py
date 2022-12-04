""" Tests for starter/model.py """
import numpy as np
import pandas as pd
import sklearn
import starter.starter.ml.data as data
import starter.starter.ml.model as model
from pytest import approx


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
    assert y_pred.shape == y.shape


def test_inference_with_encoding(test_df):
    """ Test inference loading encoders and models.
    NOTE: this requires saved encoders.
    """
    # TODO: include encoders for testing?
    data = {
        "age": 53,
        "workclass": "Federal-gov",
        "fnlwgt": 8281,
        "education": "Doctorate",
        "education_num": 891213,
        "marital_status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 1500,
        "capital_loss": 500,
        "hours_per_week": 30,
        "native_country": "Jamaica"
    }
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    #df = pd.DataFrame.from_dict(data)
    mod = model.get_model(mode="load")
    y_num = model.inference_pd(mod, df)
    print(f"y_num: {y_num}")
    y_cat = model.inference_pd(mod, df, decode=True)
    print(f"y_cat: {y_cat}")


