"""
Name: Siddhant Mahajani
Program: PredictionModel.py
Description: This is the model will help us predict performance of the player
Date: 30th Nov 2022
"""

# !/usr/bin/env python

# imported required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as obj
import plotly
import json
import ConnectionModel as conn

# imported warnings library to ignore all the warnings displayed
import warnings

warnings.filterwarnings('ignore')


# method to fetch appropriate features depending on the player position
def get_features():
    goalkeeping_features = ['overall', 'potential', 'physic', 'mentality_penalties', 'goalkeeping_diving',
                            'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
                            'goalkeeping_reflexes']

    midfield_features = ['overall', 'potential', 'physic', 'mentality_penalties', 'pace', 'shooting', 'passing',
                         'dribbling', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
                         'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
                         'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration',
                         'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
                         'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
                         'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
                         'mentality_vision', 'mentality_composure']

    defender_features = ['overall', 'potential', 'physic', 'mentality_penalties', 'mentality_composure',
                         'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
                         'movement_reactions', 'movement_balance', 'defending', 'mentality_aggression',
                         'mentality_interceptions', 'mentality_positioning', 'power_strength']

    attacking_features = ['overall', 'potential', 'physic', 'mentality_penalties', 'pace', 'shooting', 'dribbling',
                          'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                          'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina',
                          'attacking_finishing',
                          'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
                          'skill_dribbling',
                          'skill_curve', 'skill_fk_accuracy', 'skill_ball_control', 'power_strength',
                          'mentality_positioning']
    return attacking_features


# prediction method to perform player prediction
def predict_player_performance(data, features):
    # features = ['overall']
    # sort the dataframe as per age of player in descending order
    data.sort_values(by=['age'], ascending=False)
    # create an updated data frame from the data that is sent with appropriate player attributes
    updated_data = pd.DataFrame(index=range(0, len(data)), columns=[features])

    # iterate the data frame and store all the data in updated data frame
    for i in range(0, len(data)):
        updated_data[features][i] = int(data[features][i])

    # create a new variable with dataset and store all the values from updated data to dataset
    dataset = updated_data.values
    # create training and test dataset
    train = dataset[0:len(dataset) - 2, :]
    test = dataset[len(dataset) - 2:, :]

    # define a min max scaler object with feature range of 0, 1
    # the range will scale the data according to the given input. The range is either 0 to 1 or -1 to 1
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    # fit and transform the dataset using min max scaler into a scaled dataset
    scaled_data = min_max_scaler.fit_transform(dataset)

    # create a new x_train variable with the dataset
    x_train = [scaled_data]

    # create a new y_train variable with the dataset
    y_train = [scaled_data]

    # convert both the variables into a numpy array
    x_train, y_train = np.array(x_train), np.array(y_train)

    # define a Sequential model
    model = Sequential()
    # add Long-Short Term Memory (LSTM) to the model to perform prediction
    # units input is used to create the dimensionality of the output space
    model.add(LSTM(units=50))
    # add Dense to the model
    # this model is used to change the dimension of the vector
    model.add(Dense(1))

    # compile the model with the loss of mean_squared_error and optimizer as adam
    # mean_squared_error: avg of square of the difference between observed and predicted value
    # adam is a descent method based on adaptive estimation
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit the training data to the model with appropriate inputs
    # epochs: number of times the algorithm will work through the training data set
    # verbose: to print the logging errors on the screen. 0 is silent, 1 will show animated progress bar
    # 2 will just mention the line number of epoch
    model.fit(x_train, y_train, epochs=10, batch_size=5, verbose=1)

    # create a test dataset
    X_test = [dataset]

    # converted the test dataset into a numpy array with type float
    X_test = np.asarray(X_test).astype('float32')

    # performed prediction on the data present using predict method
    predicted_score = model.predict(X_test)
    # it undoes the scaling of X according to the feature range
    predicted_score = min_max_scaler.inverse_transform(predicted_score)

    # returned single predicted score of the feature used
    return predicted_score


# method to call prediction model
def perform_prediction(dataframe):
    # features that are to be used to perform prediction based on the player data
    features = get_features()
    # array to score actual scores
    original_scores = []
    # array to score predicted scores
    predicted_scores = []
    # iterate features to fetch the predicted value of each feature
    for f in features:
        original_scores.append(dataframe[f][0])
        # print(f"Performing prediction of {f}")
        predicted_score = predict_player_performance(dataframe, f)
        predicted_scores.append(predicted_score[0][0])

    # display stacked bar chart to display the predicted and original scores
    fig = obj.Figure(data=[
        obj.Bar(name='Original Scores', x=features, y=original_scores),
        obj.Bar(name='Predicted Scores', x=features, y=predicted_scores)
    ])
    fig.update_layout(barmode='group')
    fig.show()
    return fig


# main method
def predict(name):
    # fetch the data from the database
    df = conn.get_data_for_prediction()
    # perform prediction and take predictive analysis figure as output
    figure = perform_prediction(df)
    # convert the received figure into json object to display it on the UI
    graph_json = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    # print(graph_json)
    # returned the json figure
    return df['long_name'][0], graph_json
