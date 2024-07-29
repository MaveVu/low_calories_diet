from pre_process import knn_imputation
from feature_select import feature_select
from modelling import linear_model, tree_model

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv("train_data.csv")

print("Running KNN imputation")
train_data = knn_imputation(train_data)

print("Running feature selection")
selected_features = feature_select(train_data)

print("Running regression models")
selected_data = train_data[selected_features]
linear_model(selected_data)
tree_model(selected_data)
