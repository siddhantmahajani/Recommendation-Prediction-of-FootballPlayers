# Recommendation-Prediction-of-FootballPlayers

The program will recommend a football player as a replacement for the input provided and then predict his performance based on the previous data collected for the players.
The program takes player name as an input and then depending on his position, recommends the best 10 players that can replace him by identifying similarity and then based on the selected player, shows a graph of the growth of the players performance in all the modules appropriately.

Technologies used: Python, Pandas, Keras, LSTM, Cosine Similarity, Plotly

Cosine Similarity:
To recommend the best player that can replace the existing player, the algorithm calculates the cosine similarity between the players and the best players that can replace him will be displayed in ascending order of the similarity scores.

Prediction:
To predict the performance of the player LSTM is used as the data is time-series data and to predict the future score using time-series data, LSTM is the best suited and most accurate model. 

Libraries used for prediction:

    pandas: to read csv passed as input data.
    keras: to prepare appropriate models to perform the task.
    plotly: to display the prediction graph of the player.

Code: The code contains 3 files:

    RecommendationModel.py: This model will take player name as input and provide best suited replacements in a tabular format.
    PredictionModel.py: This model will provide the predicted performance of the player that is selected from the recommended players list.
    ConnectionModel.py: This model will establish connection with the files and read data from the files.

The application was a cloud based application using pyspark to connect to Apache Hive database which consisted of all the data and hosted on Google Cloud Platform but due to resource exhaustion, the code is removed and now the models run locally on the local dataset. 
