import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.tree import plot_tree


def linear_model(df: pd.DataFrame):
    print("Linear model passed df: ", df)
    X = df.drop('Energy with dietary fibre, equated (kJ)', axis=1)
    y = df['Energy with dietary fibre, equated (kJ)']
    preliminary_analyse(X, y)

    linear_model1 = ['Nitrogen (g)', 'Total saturated fatty acids, equated (%T)',
                     'Total monounsaturated fatty acids, equated (%T)',
                     'C18:2w6 (%T)', 'C16 (g)']
    linear_model2 = ['Total saturated fatty acids, equated (%T)',
                     'Total monounsaturated fatty acids, equated (%T)',
                     'C18:2w6 (%T)']
    linear_model_list = [linear_model1, linear_model2]
    for i in [0, 1]:
        X = X[linear_model_list[i]]
        residuals = linear_regress(X, y)
        residual_plot(X, residuals, f'residual_model{i}.png')


def tree_model(df):
    X = df.drop('Energy with dietary fibre, equated (kJ)', axis=1)
    y = df['Energy with dietary fibre, equated (kJ)']
    optimal_tree_depth(X, y)
    reg_tree = DecisionTreeRegressor(max_depth=5)
    reg_tree.fit(X, y)
    plt.figure(figsize=(60, 20))
    plot_tree(reg_tree, feature_names=X.columns, filled=True, rounded=True, fontsize=10, proportion=True, precision=2)
    plt.title('Decision Tree', fontsize=50)
    plt.tight_layout(pad=0.5)
    plt.savefig('regression_tree.png', dpi=100)


def linear_regress(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    return residuals


def preliminary_analyse(X, y):
    residuals = []
    size = (len(X.columns) + 2) // 3
    fig, axs = plt.subplots(size, 3, figsize=(25, 25))
    for i, col in enumerate(X.columns):
        slope, intercept = np.polyfit(X[col], y, 1)
        y_hat = slope * np.array(X[col]) + intercept
        residuals.append(np.array(y) - y_hat)
        axs[i // 3, i % 3].scatter(X[col], y, alpha=0.5)
        axs[i // 3, i % 3].plot(X[col], y_hat, color='red', label='Fitted Line')
        axs[i // 3, i % 3].set_xlabel(col)
        axs[i // 3, i % 3].set_ylabel('Energy with dietary fibre, equated (kJ)')
        axs[i // 3, i % 3].set_title(f'{col} vs Energy with dietary fibre, equated (kJ)')
    plt.tight_layout()
    plt.savefig('features_vs_energy.png')

    fig, axs = plt.subplots(size, 3, figsize=(25, 25))
    for i, col in enumerate(X.columns):
        axs[i // 3, i % 3].scatter(X[col], residuals[i], alpha=0.5)
        axs[i // 3, i % 3].axhline(y=0, color='red', linestyle='--') 
        axs[i // 3, i % 3].set_xlabel(col)
        axs[i // 3, i % 3].set_ylabel('Residuals')
        axs[i // 3, i % 3].set_title(f'Residual Plot for {col}')
    plt.tight_layout()
    plt.savefig('residual_preliminary.png')


def residual_plot(X, residuals, filename):
    num_cols = X.shape[1]
    size = (num_cols + 1) // 2
    fig, axs = plt.subplots(size, min(num_cols, 2), figsize=(25, 25))
    axs = axs.ravel()
    for i, col in enumerate(X.columns):
        axs[i].scatter(X[col], residuals, alpha=0.5)
        axs[i].axhline(y=0, color='red', linestyle='--')
        axs[i].set_xlabel(col, fontsize=22)
        axs[i].set_ylabel('Residuals', fontsize=22)
        axs[i].set_title(f'Residual Plot for {col}', fontsize=28)
        axs[i].tick_params(axis='both', labelsize=20)
        axs[i].title.set_size(24)
    for j in range(num_cols, size*2):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.savefig(filename)


def optimal_tree_depth(X, y):
    cv_mean = []
    depths = range(1, 10)
    for depth in depths:
        tree_model = DecisionTreeRegressor(max_depth=depth)
        mse_scores = -cross_val_score(tree_model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_mean.append(mse_scores.mean())
    cv_scores_mean = np.array(cv_mean)
    print(cv_scores_mean)
    plot_tree_depth(depths, cv_scores_mean)
    optimal_depth = np.argmax(cv_scores_mean) + 1
    return optimal_depth


def plot_tree_depth(depths, cv_scores_mean):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(depths, cv_scores_mean, '-o', label='5-fold CV MSE (scaled)', alpha=0.9)
    ylim = plt.ylim()
    ax.set_title('Accuracy for different tree depths', fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('CV Score', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()
    plt.tight_layout()
    plt.savefig('tree_depth_score.png')
