# Script to train machine learning model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import data, model
from starter.starter.ml.model import compute_model_metrics

# load data as dataframe
DATA_PATH = "../../data/census.csv"
df = data.load_data(DATA_PATH)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

# train data
cat_features = data.get_categorical_features()
LABEL = "salary"
X_train, y_train, encoder, lb = data.process_data(
    data=train, categorical_features=cat_features, label=LABEL, training=True
)

# Proces the test data with the process_data function.
# get mapping for cols; this way we can avoid processing each slice separately
mapping_cols = data.get_mapping_cols(test, label=LABEL)

X_test, y_test, encoder, lb = data.process_data(
    data=test,
    categorical_features=cat_features,
    label=LABEL,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model
trained_model = model.train_model(X_train, y_train)

# performance
precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, model.inference(trained_model, X_test))
