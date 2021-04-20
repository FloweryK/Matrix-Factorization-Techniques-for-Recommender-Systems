import os
import requests
import pandas as pd


def read_movielens():
    df = pd.read_csv('data/ml-latest-small/ratings.csv')
    df.columns = ['userId', 'itemId', 'rating', 'timestamp']
    df = df.sort_values(by=['timestamp'])
    return df


def read_amazon(category):
    # available categories: see https://nijianmo.github.io/amazon/index.html#subsets
    # Grocery_and_Gourmet_Food, AMAZON_FASHION.csv
    path = f'data/amazon-{category}/{category}.csv'
    url = f'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/{category}.csv'

    # check whether the data is already downloaded or not
    if not os.path.exists(path):
        print(f'amazon dataset {category} not found.')
        print(f'downloading dataset {category} from {url}')
        os.makedirs(f'data/amazon-{category}', exist_ok=True)
        with open(path, 'wb') as f:
            f.write(requests.get(url).content)

    df = pd.read_csv(path)
    df.columns = ['itemId', 'userId', 'rating', 'timestamp']
    df = df.sort_values(by=['timestamp'])
    return df

