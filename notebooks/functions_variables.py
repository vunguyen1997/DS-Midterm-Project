#import necessary libraries
import pandas as pd
import numpy as np
import os
import json

def encode_tags(df):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    tags = df["tags"].tolist()
    # create a unique list of tags and then create a new column for each tag
        
    return df

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

# fill null values in columns with 0
def fill_column_null_values(dataframe, columns_fill):
    for col in columns_fill:
        dataframe.loc[:, col] = dataframe[col].fillna(0)
    filled_df = dataframe
    return filled_df