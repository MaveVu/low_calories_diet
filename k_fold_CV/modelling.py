import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


def linear_model(train_data: pd.DataFrame, test_data: pd.DataFrame, linear_features: list):
    for feature in linear_features:
        if feature not in test_data or feature not in train_data:
            print("Can't found feature ", feature)
            print("Can't fit linear model with selected features on this fold.")
            return None

    X_train = train_data[linear_features]
    y_train = train_data['Energy with dietary fibre, equated (kJ)']
    X_test = test_data[linear_features]
    y_test = test_data['Energy with dietary fibre, equated (kJ)']

    model = LinearRegression()
    model.fit(X_train, y_train)
    intercept = model.intercept_
    coefficients = model.coef_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    scaled_mse = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return intercept, coefficients, [r2_train, r2_test, scaled_mse]


def tree_model(train_data: pd.DataFrame, test_data: pd.DataFrame):
    X_train = train_data.drop('Energy with dietary fibre, equated (kJ)', axis=1)
    y_train = train_data['Energy with dietary fibre, equated (kJ)']
    X_test = test_data.drop('Energy with dietary fibre, equated (kJ)', axis=1)
    y_test = test_data['Energy with dietary fibre, equated (kJ)']

    reg_tree = DecisionTreeRegressor(max_depth=5)
    reg_tree.fit(X_train, y_train)
    y_test_pred = reg_tree.predict(X_test)
    scaled_mse = mean_squared_error(y_test, y_test_pred)

    return scaled_mse
