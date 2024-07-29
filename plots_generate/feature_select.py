import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold


def feature_select(df: pd.DataFrame):
    selector = VarianceThreshold()
    selector.fit(df)
    non_zero_indices = selector.get_support(indices=True)
    non_zero_features = df.columns[non_zero_indices]
    df = df[non_zero_features]
    df = normal_transform(df)

    # Discretization & NMI using Equal Width Binning
    binned_df = binning(df, 'uniform')
    corr_matrix = nmi_matrix(binned_df)
    corr_heatmap(corr_matrix, 'before_width_bin.png', (25, 25), 10)
    new_corr_matrix = feature_remove(corr_matrix)
    scatter_matrix(df[new_corr_matrix.columns], 'width_scatter_matrix.png')

    # Discretization & NMI using Equal Frequency Binning
    binned_df = binning(df, 'quantile')
    corr_matrix = nmi_matrix(binned_df)
    corr_heatmap(corr_matrix, 'before_freq_bin.png', (25, 25), 10)
    new_corr_matrix = feature_remove(corr_matrix)
    corr_heatmap(new_corr_matrix, 'after_freq_bin.png', (15, 15), 20)
    scatter_matrix(df[new_corr_matrix.columns], 'freq_scatter_matrix.png')
    selected_features = new_corr_matrix.columns

    # Feature Selection using Pearson
    corr_matrix = df.corr()
    corr_matrix = corr_matrix.astype(float).round(1)
    corr_heatmap(corr_matrix, 'before_pearson.png', (25, 25), 10)
    new_corr_matrix = feature_remove(corr_matrix)
    scatter_matrix(df[new_corr_matrix.columns], 'pearson_scatter_matrix.png')

    return selected_features


def binning(df: pd.DataFrame, bin_strategy: str):
    binned_df = df
    binned_df = binned_df.reset_index(drop=True)
    equal_width = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=bin_strategy)
    for col in df.columns:
        binned_df[col] = equal_width.fit_transform(df[col].values.reshape(-1, 1)).astype(int)
    return binned_df


def nmi_matrix(binned_df: pd.DataFrame):
    matrix = pd.DataFrame(index=binned_df.columns, columns=binned_df.columns)
    num_cols = binned_df.shape[1]
    for col in range(num_cols):
        for row in range(col, num_cols):
            nmi = normalized_mutual_info_score(binned_df.iloc[:, col],
                                               binned_df.iloc[:, row], average_method='min')
            matrix.iloc[row, col] = matrix.iloc[col, row] = nmi
    return round(matrix.astype(float), 1)


def corr_heatmap(corr_matrix: pd.DataFrame, filename: str, sfig: tuple, sfont: int, title: str):
    corr_matrix = corr_matrix.drop('Energy with dietary fibre, equated (kJ)', axis=0)
    corr_matrix = corr_matrix.drop('Energy with dietary fibre, equated (kJ)', axis=1)
    fig, ax = plt.subplots(figsize=sfig)
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", linewidths=.5, annot_kws={"fontsize":sfont-2})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)


def feature_remove(corr_matrix: pd.DataFrame):
    thres0 = 0.1
    thres1 = 0.5
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for col in upper.columns:
        if abs(upper[col]['Energy with dietary fibre, equated (kJ)']) <= thres0 \
           or any(abs(upper[col]) >= thres1):
            to_drop.append(col)
    corr_matrix = corr_matrix.drop(corr_matrix[to_drop], axis=0)
    corr_matrix = corr_matrix.drop(corr_matrix[to_drop], axis=1)
    return corr_matrix


def scatter_matrix(df: pd.DataFrame, filename: str):
    df = df.drop('Energy with dietary fibre, equated (kJ)', axis=1)
    fig, ax = plt.subplots(figsize=(25, 25))
    pd.plotting.scatter_matrix(df, alpha=0.6, ax=ax)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.savefig(filename)


def normal_transform(df: pd.DataFrame):
    left_thres = -1
    right_thres = 1
    pt = PowerTransformer(method='yeo-johnson')
    for col in df.columns:
        skewness = df[col].skew()
        if skewness < left_thres or skewness > right_thres:
            df.loc[:, col] = pt.fit_transform(df.loc[:, col].values.reshape(-1, 1))
    return df
