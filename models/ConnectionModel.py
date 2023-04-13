"""
Name: Siddhant Mahajani
Program: ConnectionModel.py
Description: This is the model will help read data from the files
Date: 30th Nov 2022
"""

# !/usr/bin/env python

# import libraries required to establish connection
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType


# method to create schema as spark session requires schema to fetch headers from the file
"""def get_player_schema():
    player_schema = StructType().add("short_name", "string").add("long_name", "string").add("player_positions",
                                                                                            "string") \
        .add("overall", "string").add("potential", "string").add("value_eur", "string").add("wage_eur", "string") \
        .add("age", "string").add("dob", "string").add("height_cm", "string").add("weight_kg", "string") \
        .add("club_team_id", "string").add("club_name", "string").add("league_name", "string") \
        .add("league_level", "string").add("club_position", "string").add("club_jersey_number", "string") \
        .add("club_loaned_from", "string").add("club_joined", "string").add("club_contract_valid_until", "string") \
        .add("nationality_id", "string").add("nationality_name", "string").add("nation_team_id", "string") \
        .add("nation_position", "string").add("nation_jersey_number", "string").add("preferred_foot", "string") \
        .add("weak_foot", "string").add("skill_moves", "string").add("international_reputation", "string") \
        .add("work_rate", "string").add("body_type", "string").add("release_clause_eur", "string") \
        .add("player_tags", "string").add("player_traits", "string").add("pace", "string").add("shooting", "string") \
        .add("passing", "string").add("dribbling", "string").add("defending", "string").add("physic", "string") \
        .add("attacking_crossing", "string").add("attacking_finishing", "string") \
        .add("attacking_heading_accuracy", "string").add("attacking_short_passing", "string") \
        .add("attacking_volleys", "string").add("skill_dribbling", "string").add("skill_curve", "string") \
        .add("skill_fk_accuracy", "string").add("skill_long_passing", "string").add("skill_ball_control", "string") \
        .add("movement_acceleration", "string").add("movement_sprint_speed", "string") \
        .add("movement_agility", "string").add("movement_reactions", "string").add("movement_balance", "string") \
        .add("power_shot_power", "string").add("power_jumping", "string").add("power_stamina", "string") \
        .add("power_strength", "string").add("power_long_shots", "string").add("mentality_aggression", "string") \
        .add("mentality_interceptions", "string").add("mentality_positioning", "string") \
        .add("mentality_vision", "string").add("mentality_penalties", "string").add("mentality_composure", "string") \
        .add("defending_marking_awareness", "string").add("defending_standing_tackle", "string") \
        .add("defending_sliding_tackle", "string").add("goalkeeping_diving", "string") \
        .add("goalkeeping_handling", "string").add("goalkeeping_kicking", "string") \
        .add("goalkeeping_positioning", "string").add("goalkeeping_reflexes", "string") \
        .add("goalkeeping_speed", "string").add("ls", "string").add("st", "string").add("rs", "string") \
        .add("lw", "string").add("lf", "string").add("cf", "string").add("rf", "string").add("rw", "string") \
        .add("lam", "string").add("cam", "string").add("ram", "string").add("lm", "string").add("lcm", "string") \
        .add("cm", "string").add("rcm", "string").add("rm", "string").add("lwb", "string").add("ldm", "string") \
        .add("cdm", "string").add("rdm", "string").add("rwb", "string").add("lb", "string").add("lcb", "string") \
        .add("cb", "string").add("rcb", "string").add("rb", "string").add("gk", "string") \
        .add("player_face_url", "string").add("club_logo_url", "string").add("club_flag_url", "string") \
        .add("nation_logo_url", "string").add("nation_flag_url", "string")
    return player_schema
"""

"""
# as there are multiple files, this will create a list of all the files along with the path
def get_file_paths():
    file_names = ['players_fifa_15_clean_final.txt', 'players_fifa_16_clean_final.txt',
                  'players_fifa_17_clean_final.txt', 'players_fifa_18_clean_final.txt',
                  'players_fifa_19_clean_final.txt', 'players_fifa_20_clean_final.txt',
                  'players_fifa_21_clean_final.txt', 'players_fifa_22_clean_final.txt']
    file_paths = []
    for name in file_names:
        file_paths.append("hdfs://localhost:9000/user/user_1/" + name)

    return file_paths


# creating session using spark with hadoop on local host
def create_session():
    session = SparkSession.builder.master("local").appName("player_recommendation_prediction").getOrCreate()
    return session


# method to return data that is required to perform recommendation
def get_data_for_recommendation():
    session = create_session()
    schema = get_player_schema()
    recommendation_file_path = "hdfs://localhost:9000/user/user_1/players_fifa_22_clean_final.txt"
    data = session.read.csv(recommendation_file_path, schema)
    # convert the data fetched by spark session to pandas data frame
    data_frame = data.toPandas()
    # removed all the white spaces from the short name to perform accurate data filtering
    data_frame['short_name'] = data_frame['short_name'].str.replace(" ", "")

    return data_frame
"""


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
