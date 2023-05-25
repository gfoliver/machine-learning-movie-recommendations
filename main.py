import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the MovieLens dataset into DataFrames
ratings_data = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')

# Data preprocessing
user_mapping = {id: i for i, id in enumerate(ratings_data['userId'].unique())}
reverse_user_mapping = {v: k for k, v in user_mapping.items()}
movie_mapping = {id: i for i, id in enumerate(ratings_data['movieId'].unique())}
reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}
ratings_data['userId'] = ratings_data['userId'].map(user_mapping)
ratings_data['movieId'] = ratings_data['movieId'].map(movie_mapping)

# Split the data into training and testing sets
train_data, test_data = train_test_split(ratings_data, test_size=0.2, random_state=42)

# Validate the mapping between movie IDs and indices
max_movie_id = max(movie_mapping.values())
if max_movie_id >= len(movie_mapping):
    raise ValueError("Movie ID mapping is incorrect. Please check the movieId column in the ratings.csv file.")

# Define model architecture using Keras API
num_users = len(user_mapping)
num_movies = len(movie_mapping)
embedding_dim = 30

user_input = tf.keras.layers.Input(shape=(1,))
movie_input = tf.keras.layers.Input(shape=(1,))

user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim, input_length=1)(user_input)
movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_dim, input_length=1)(movie_input)

user_embedding = tf.keras.layers.Flatten()(user_embedding)
movie_embedding = tf.keras.layers.Flatten()(movie_embedding)

concat = tf.keras.layers.Concatenate()([user_embedding, movie_embedding])
dense = tf.keras.layers.Dense(64, activation='relu')(concat)
output = tf.keras.layers.Dense(1)(dense)

model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([train_data['userId'], train_data['movieId']], train_data['rating'], epochs=10, batch_size=64)

# Make predictions on test data
predictions = model.predict([test_data['userId'], test_data['movieId']])

# Print recommendations for each user
user_id = int(input("Type in the userId: "))

if user_id not in user_mapping:
    raise ValueError("User doesnt exist")

user_id = user_mapping[user_id]

user_ratings = test_data[test_data['userId'] == user_id]
unrated_movies = movies_data[~movies_data['movieId'].isin(user_ratings['movieId'])]

# Validate the movie IDs in unrated_movies DataFrame
unrated_movie_ids = unrated_movies['movieId'].unique()
valid_unrated_movie_ids = [id for id in unrated_movie_ids if id in movie_mapping]

if not valid_unrated_movie_ids:
    raise ValueError(f"No recommendations available for User ID: {reverse_user_mapping[user_id]}")


user_indices = np.full(len(valid_unrated_movie_ids), user_id)
movie_indices = np.array([movie_mapping[id] for id in valid_unrated_movie_ids])
predicted_ratings = model.predict([user_indices, movie_indices])

# Combine movie IDs, titles, and predicted ratings into a DataFrame
recommendations = pd.DataFrame({
    'movieId': valid_unrated_movie_ids,
    'predicted_rating': predicted_ratings.flatten()
})
recommendations = recommendations.merge(movies_data, on='movieId')

# Sort the recommendations by predicted rating in descending order
recommendations = recommendations.sort_values('predicted_rating', ascending=False)

print(f"Recommendations for User ID: {reverse_user_mapping[user_id]}")
for _, movie in recommendations.head(5).iterrows():
    movie_id = movie['movieId']
    if movie_id in reverse_movie_mapping:
        print(
            f"Movie ID: {reverse_movie_mapping[movie_id]} - Title: {movie['title']} - Predicted Rating: {movie['predicted_rating']}")
    else:
        print(
            f"Movie ID: {movie_id} - Title: {movie['title']} - Predicted Rating: {movie['predicted_rating']} (Movie ID not found in mapping)")