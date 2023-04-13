"""
Name: Siddhant Mahajani
Program: ConnectionModel.py
Description: This is the model will help read data from the files
Date: 30th Nov 2022
"""

# !/usr/bin/env python

# import libraries required to establish connection
import pandas as pd


# as there are multiple files, this will create a list of all the files along with the path
def get_file_paths():
    file_names = ['players_15.csv', 'players_16.csv',
                  'players_17.csv', 'players_18.csv',
                  'players_19.csv', 'players_20.csv',
                  'players_21.csv', 'players_22.csv']
    file_paths = []
    for name in file_names:
        file_paths.append("/Users/******/Documents/Football_dataset/" + name)

    return file_paths


# method to return data that is required to perform recommendation
def get_data_for_recommendation():
    # session = create_session()
    # schema = get_player_schema()
    recommendation_file_path = "/Users/******/Documents/Football_dataset/players_22.csv"
    # convert the data fetched by spark session to pandas data frame
    data_frame = pd.read_csv(recommendation_file_path)
    # removed all the white spaces from the short name to perform accurate data filtering
    data_frame['short_name'] = data_frame['short_name'].str.replace(" ", "")

    return data_frame


# method to return data required to perform predictions from all the files
def get_data_for_prediction(name):
    # session = create_session()
    # schema = get_player_schema()

    # prediction_file_paths = get_file_paths()
    data_frame = pd.read_csv('/Users/******/Documents/Football_dataset'
                             '/data.csv')
    # convert the data fetched by spark session to pandas data frame
    # data_frame = data.toPandas()
    # removed all the white spaces from the short name to perform accurate data filtering
    data_frame['short_name'] = data_frame['short_name'].str.replace(" ", "")

    updated_data_frame = data_frame[data_frame['short_name'] == name]

    return updated_data_frame
