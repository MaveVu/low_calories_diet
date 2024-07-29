from pre_process import pre_process, knn_imputation, to_drop_null
from feature_select import feature_select
from modelling import linear_model, tree_model

import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict
import numpy as np

import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('nutrient-file-release2-jan22.xlsx', sheet_name=1)

df = pre_process(df)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
intercept_list = defaultdict(list)
coefficients_list = defaultdict(list)
linear_scores_list = defaultdict(list)
tree_mse_list = []
for train_index, test_index in kf.split(df):
    train_data = df.iloc[train_index].dropna(how='all')
    test_data = df.iloc[test_index].dropna(how='all')

    to_drop = list(set(to_drop_null(train_data) + to_drop_null(test_data)))
    train_data = train_data.drop(to_drop, axis=1)
    test_data = test_data.drop(to_drop, axis=1)

    train_data = knn_imputation(train_data)
    test_data = knn_imputation(test_data)

    linear_model1 = ['Nitrogen (g)', 'Total saturated fatty acids, equated (%T)',
                     'Total monounsaturated fatty acids, equated (%T)',
                     'C18:2w6 (%T)', 'C16 (g)']
    linear_model2 = ['Total saturated fatty acids, equated (%T)',
                     'Total monounsaturated fatty acids, equated (%T)',
                     'C18:2w6 (%T)']
    linear_model_list = [linear_model1, linear_model2]
    for i in [0, 1]:
        if linear_model(train_data, test_data, linear_model_list[i]):
            intercept, coefficients, scores \
                = linear_model(train_data, test_data, linear_model_list[i])
            intercept_list[i].append(intercept)
            coefficients_list[i].append(coefficients)
            linear_scores_list[i].append(scores)

    selected_features = feature_select(train_data)
    train_data = train_data[selected_features]
    test_data = test_data[selected_features]

    tree_scaled_mse = tree_model(train_data, test_data)
    tree_mse_list.append(tree_scaled_mse)

goodness_df = pd.DataFrame(columns=['10-fold CV Train R-squared',
                                    '10-fold CV Test R-squared',
                                    '10-fold CV MSE (scaled)'])

for i in [0, 1]:
    avg_intercept = np.array(intercept_list[i]).mean()
    avg_coefficients = np.mean(coefficients_list[i], axis=0)
    avg_linear_scores = np.mean(linear_scores_list[i], axis=0)

    goodness_df.loc[f'Linear model {i + 1}'] = np.round(avg_linear_scores.astype(float), 3)
    print(f'Model {i + 1}\'s avg line intercept: ', avg_intercept)
    print(f'Model {i + 1}\'s avg line coefficients: ', avg_coefficients)

avg_tree_mse = np.array(tree_mse_list).mean()

goodness_df.loc['Regression Tree'] = ['n/a', 'n/a', np.round(avg_tree_mse.astype(float), 3)]

goodness_df.to_csv('goodness_of_fit.csv', index=True)
