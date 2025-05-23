#import necessary libraries
import pandas as pd
import numpy as np
import os
from collections import Counter
import json

# Filters out tags from the datataframe that appear less than count_num amount of times
def remove_infrequent_tags(dataframe, column, count_num):
    # Flatten list values, skipping nulls
    flat_tags = [item for sublist in dataframe[column].dropna() for item in sublist]

    # Count using Counter
    tag_counts = Counter(flat_tags)
    
    # Filter tags that appear less than count_num times
    tags_to_remove = {tag for tag, count in tag_counts.items() if count < count_num}

    # getting rid of tags that are under count_num frequency
    dataframe.loc[:,column] = dataframe[column].apply(
        lambda x: [tag for tag in x if tag not in tags_to_remove] if isinstance(x, list) else x
    )
    new_df = dataframe
    return new_df

# Filter the dataframe column you want to get the top tags out of, optional arguement to allow you to additionally drop certain tags from the top list incase you want to remove redundant/unimportant tags
def get_important_tags(dataframe, column, droplist=None):
    # Count all tags across rows, skipping nulls
    tag_counts = Counter(tag for tags in dataframe[column].dropna() for tag in tags)
    
    # Get top 20 most common tags
    common_tags = [tag for tag, count in tag_counts.most_common(20)]
    
    # If a droplist is provided, filter those out
    if droplist is not None:
        common_tags = [tag for tag in common_tags if tag not in droplist]

    return common_tags


# Function takes the tags from important_tags that appear in column_to_encode and encodes the tags in the dataframe, then optionally drops the initial tags column if you choose to
def encode_tags(dataframe, column_to_encode, important_tags, drop_column=False):
    for tag in important_tags:
        dataframe[f'tag_{tag}'] = dataframe[column_to_encode].apply(
        lambda x: int(tag in x) if isinstance(x, list) else 0
    )
    if drop_column:
        dataframe = dataframe.drop(column_to_encode, axis=1)

    return dataframe  
    

# Extract house data from all JSON files in data folder
def json_extraction(direc='../data'):
    results = []
    count = 0
    for file_name in os.listdir(direc):
        if file_name.endswith(".json"):
            json_path = os.path.join(direc, file_name)
            with open(json_path) as f:
                file_data = json.load(f)
            data = file_data['data']['results']
            results.append(data)
            count+=1
    return results

results = json_extraction()

def normalized_dataframe(results):
    # create empty data_set
    df_set = []
    # loop cycles through each result individual normalization
    for i in range(len(results)):
        df = pd.json_normalize(results[i])
        df_set.append(df)
    df = pd.concat(df_set)
    return df

def del_columns(dataframe,drop_attributes):
    refined_df = dataframe.drop(columns= drop_attributes)
    return refined_df

# drop null values in columns
def del_column_null_values(dataframe, drop_column_rows):
    new_df = dataframe.dropna(subset=drop_column_rows)
    return new_df

# fill null values in columns with anoter value
def fill_column_null_values(dataframe, columns_fill, value=0):
    for col in columns_fill:
        dataframe.loc[:, col] = dataframe[col].fillna(value)
    filled_df = dataframe
    return filled_df

# develop functions for Part 3
import itertools
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def custom_cross_validation(training_data, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_folds, val_folds = [], []
    for train_idx, val_idx in kf.split(training_data):
        train_folds.append(training_data.iloc[train_idx])
        val_folds.append(  training_data.iloc[val_idx])
    return train_folds, val_folds

def hyperparameter_search(training_folds, validation_folds, param_grid):
    # 1) Generate hyperparam combos
    param_names   = list(param_grid.keys())
    hyperparams   = list(itertools.product(*param_grid.values()))
    
    hyperparam_scores = []  # store avg RMSE for each combo

    # 2) Loop through each hyperparam combination
    for combo in hyperparams:
        params = dict(zip(param_names, combo))
        scores = []  # one RMSE per fold

        # 3) Loop through each fold
        for train_df, val_df in zip(training_folds, validation_folds):
            # assume last column is target
            target = train_df.columns[-1]
            X_tr, y_tr = train_df.drop(target, axis=1), train_df[target]
            X_va, y_va = val_df.drop(target, axis=1), val_df[target]

            model = RandomForestRegressor(random_state=42, **params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)

            rmse = np.sqrt(mean_squared_error(y_va, preds))
            scores.append(rmse)

        # average across folds
        avg_score = np.mean(scores)
        hyperparam_scores.append(avg_score)

    # 4) Find the index of the best (lowest) RMSE
    best_idx    = int(np.argmin(hyperparam_scores))
    best_params = dict(zip(param_names, hyperparams[best_idx]))

    return best_params

