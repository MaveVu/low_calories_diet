import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


def pre_process(df: pd.DataFrame):
    df.columns = [re.sub(r'\n', '', text) for text in df.columns]
    df = df.drop(['Public Food Key', 'Classification', 'Food Name',
                  'Energy, without dietary fibre, equated (kJ)',
                  'Protein (g)', 'Fat, total (g)', 'Total dietary fibre (g)',
                  'Available carbohydrate, without sugar alcohols (g)',
                  'Available carbohydrate, with sugar alcohols (g)'], axis=1)
    df = drop_null_cols(df)
    return df


def drop_null_cols(df: pd.DataFrame):
    thresh = len(df) * 2/3
    df = df.dropna(axis=1, thresh=thresh)
    return df


def knn_imputation(df: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    knn_imputer = KNNImputer()
    df = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
    return df
