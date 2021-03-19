import numpy as np


def split_dataset_by_time(df, ratio):
    df = df.sort_values(by=['timestamp'])

    # split time in data
    t_split = df['timestamp'][int(ratio * len(df))]
    df_train = df[df['timestamp'] <= t_split]
    df_future = df[(df['timestamp'] > t_split)
                   & df['userId'].isin(df_train['userId'].unique())
                   & df['itemId'].isin(df_train['itemId'].unique())].sample(frac=1)
    df_test = df_future.iloc[:int(0.5 * len(df_future))]
    df_vali = df_future.iloc[int(0.5 * len(df_future)):]

    return df_train, df_test, df_vali


def make_matrix(df, all_columns):
    mat = df.pivot(index='userId', columns='itemId', values='rating')
    mat[np.setdiff1d(all_columns, mat.columns)] = np.nan
    mat = mat.fillna(0)
    mat = mat.sort_index(axis=0)
    mat = mat.sort_index(axis=1)
    return mat