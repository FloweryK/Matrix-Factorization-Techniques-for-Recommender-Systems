def split_df_by_time(df, ratio):
    t_split = df['timestamp'].tolist()[int(len(df) * ratio)]
    df_train = df[df['timestamp'] <= t_split]
    df_future = df[(df['timestamp'] > t_split)
                   & df['userId'].isin(df_train['userId'].unique())
                   & df['itemId'].isin(df_train['itemId'].unique())].sample(frac=1)
    df_test = df_future.iloc[:int(0.5 * len(df_future))]
    df_vali = df_future.iloc[int(0.5 * len(df_future)):]
    return df_train, df_test, df_vali