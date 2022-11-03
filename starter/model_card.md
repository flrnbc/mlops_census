# Model Card

## Model Details
The model was trained by flrnbc. It is a logistic regression model using the default hyperparameters of scikit-learn 1.1.2.

## Intended Use
Given census data (education, marital-status, occupation, race, etc.), the model is supposed to predict if the corresponding citizen earns >50k or <=50k per year. In particular, it is a binary classification problem.

## Training Data
The original data is retrievable from https://archive.ics.uci.edu/ml/datasets/census+income. The original data (`data/census_orig.csv`) consists of 48842 rows. A manual inspection showed that the data was not clean. It was manually cleaned: removing whitespaces, replacing `?` with `nan` values (TODO: implemented automatic cleanup).
To split the data into train and test set, we used an 80-20 split. All the categorical features were OneHotEncoded and a label binarizer was used for the labels (income `>50K` is mapped to `1` and `<=50k` to `0` by the label binarizer). These encoders were added to the dvc repository as well.

## Evaluation Data
The model was evaluated on the test set of the above split. Since the dataset is quite large, 20 percent seemed sufficient to represent the overall statistics of the dataset. However, a future iteration of the model will use a more refined evaluation, e.g. K-fold cross validation.

## Metrics

The model was evaluated with three metrics:

+ `Precision = tp/(tp+fp)`
+ `Recall = tp/(tp+fn)`
+ `fbeta = (1+beta^2)*(precision*recall)/(beta^2*precision+recall)`

Here `tp` are the true positives, `fp` the false positives and `fn` for false negatives in the test data. The evaluation showed the following:

+ Precision: 0.717206132879046
+ Recall: 0.26918158567774936
+ fbeta: 0.3914458391445839i

Morever, the same metrics were determined on all categorical data slices. For example, for the feature `race` we obtain

|race                   |precision         |recall             |fbeta              |
|-----------------------|------------------|-------------------|-------------------|
|race=White             |0.7323420074349443|0.2743732590529248 |0.39918946301925023|
|race=Asian-Pac-Islander|0.6923076923076923|0.20454545454545456|0.3157894736842105 |
|race=Other             |0.3333333333333333|0.16666666666666666|0.2222222222222222 |
|race=Black             |0.5333333333333333|0.22535211267605634|0.31683168316831684|
|race=Amer-Indian-Eskimo|0.3333333333333333|0.14285714285714285|0.2                |

## Ethical Considerations

The metrics on the `race` data slice indicate that the model is biased towards `race=White`. This is because the metrics on that slice are very close to the metrics of the overall model. This is in stark contrast to e.g. the data slices `race=Black` and `race=Amer-Indian-Eskimo`. 

## Caveats and Recommendations

Even though the precision of the model is OK, its recall is very low. Hence it predicts a too many false negatives, i.e. predicting income `<=50K` but it is actually `>50K`, see the mapping of the label binarizer. So a closer look at the metrics of the data slices seems necessary and considering alternative binary classifiers is recommended.