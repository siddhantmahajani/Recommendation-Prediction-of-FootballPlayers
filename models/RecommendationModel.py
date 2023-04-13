"""
Name: Siddhant Mahajani
Program: RecommendationModel.py
Description: This is the model will help us recommend a replacement player for the player
Date: 30th Nov 2022
"""

# !/usr/bin/env python

# import statements required for the program
import operator

import numpy as np
from numpy.linalg import norm
import ConnectionModel as conn

import warnings

warnings.filterwarnings('ignore')


# method to read data from the csv (for time being, later the data will be fetched from the database)
def get_recommendation_data():
    dataset = conn.get_data_for_recommendation()
    return dataset


# method to fetch appropriate features depending on the player position
def get_features(position):

    if position.lower() == 'GK'.lower():
        features = ['overall', 'potential', 'physic', 'mentality_penalties', 'goalkeeping_diving',
                    'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning',
                    'goalkeeping_reflexes']

    elif position.lower() == 'CB'.lower() or position.lower() == 'LB'.lower() or position.lower() == 'RB'.lower() \
            or position.lower() == 'LCB'.lower() or position.lower() == 'RCB'.lower() or position.lower() == 'CDM'.lower() \
            or position.lower() == "RDM".lower() or position.lower() == "LDM".lower():
        features = ['overall', 'potential', 'physic', 'mentality_penalties', 'pace', 'mentality_composure',
                    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
                    'movement_reactions', 'movement_balance', 'defending', 'mentality_aggression',
                    'mentality_interceptions', 'mentality_positioning', 'power_strength']

    elif position.lower() == 'RM'.lower() or position.lower() == 'LM'.lower() or position.lower() == 'CM'.lower()\
            or position.lower() == 'LCM'.lower() or position.lower() == 'RCM'.lower()\
            or position.lower() == 'CAM'.lower() or position.lower() == 'LAM'.lower()\
            or position.lower() == 'RAM'.lower():
        features = ['overall', 'potential', 'physic', 'mentality_penalties', 'pace', 'shooting', 'passing',
                    'dribbling', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
                    'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
                    'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration',
                    'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
                    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
                    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
                    'mentality_vision', 'mentality_composure']
    else:
        features = ['overall', 'potential', 'physic', 'mentality_penalties', 'pace', 'shooting', 'dribbling',
                    'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions',
                    'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina',
                    'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
                    'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
                    'skill_ball_control', 'power_strength', 'mentality_positioning']

    return features


def get_filtered_data(data, from_price, to_price, from_age, to_age):
    data = data.loc[(from_price < data['value_eur']) & (data['value_eur'] < to_price) & (20 < data['age'])
                    & (data['age'] < 30)]
    return data


def get_player_attributes(pl_Name, data):
    player_Obj = data[data['short_name'] == pl_Name]
    return player_Obj


def calculate_cosine_similarity(player1, player2, data):
    a = player1
    b = get_player_attributes(player2, data)
    # print(a)
    # print(b)
    player1_Features = []
    player2_Features = []
    features = get_features(str(b['club_position']))
    for feature in features:
        player1_Features.append(a[feature])
        player2_Features.append(b[feature])
    cosine_dist = np.dot(player1_Features, player2_Features) / (norm(player1_Features) * norm(player2_Features))
    return cosine_dist


def get_nearest_neighbors(player_Name, data):
    distances = []
    for i, player_Id in data.iterrows():
        if not(player_Name.lower() in str(player_Id['short_name']).lower()):
            dist = calculate_cosine_similarity(player_Id, player_Name, data)
            distances.append((player_Id['long_name'], dist, player_Id['short_name']))

    player_Neighbors = []

    for i in range(len(distances)):
        if str(distances[i][1]) != 'nan':
            player_Neighbors.append(distances[i])
    return player_Neighbors


def get_recommendations(player_name):
    dataset = conn.get_data_for_recommendation()
    # filtered_data = get_filtered_data(dataset, from_price, to_price, from_age, to_age)
    neighbors = get_nearest_neighbors(player_name, dataset)
    neighbors.sort(key=operator.itemgetter(1), reverse=True)
    players_dict = {}
    for i in range(10):
        players_dict[str(neighbors[i][2])] = str(neighbors[i][0])
    return players_dict


print(get_recommendations("L. Messi"))
