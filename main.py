import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# CONSTANT USERID
USERID = 999999


def load_data():
    # Load the MovieLens dataset into DataFrames
    ratings_data = pd.read_csv('ratings.csv')
    movies_data = pd.read_csv('movies.csv')
    return ratings_data, movies_data


def prompt_user(movies_data, ratings_data):
    # prompt the user to rate 5 random movies from movies_data and save those ratings into a DataFrame
    user_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
    for _ in range(5):
        movie_id = np.random.choice(movies_data['movieId'].unique())
        # display movie title
        title = movies_data[movies_data['movieId'] == movie_id]['title'].values[0]
        genres = movies_data[movies_data['movieId'] == movie_id]['genres'].values[0]
        print(f"Please rate the movie '{title}' [{genres}]: ")
        rating = float(input())
        user_ratings = pd.concat([user_ratings, pd.DataFrame([{'userId': USERID, 'movieId': movie_id, 'rating': rating}])], ignore_index=True)

    # append the user ratings to the ratings_data DataFrame
    ratings_data = pd.concat([ratings_data, user_ratings], ignore_index=True)

    return ratings_data


def preprocess_data(ratings_data):
    # Data preprocessing
    user_mapping = {id: i for i, id in enumerate(ratings_data['userId'].unique())}
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    movie_mapping = {id: i for i, id in enumerate(ratings_data['movieId'].unique())}
    reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}
    ratings_data['userId'] = ratings_data['userId'].map(user_mapping)
    ratings_data['movieId'] = ratings_data['movieId'].map(movie_mapping)

    return ratings_data, user_mapping, reverse_user_mapping, movie_mapping, reverse_movie_mapping


def train(model, train_data):
    # Train the model
    model.fit([train_data['userId'], train_data['movieId']], train_data['rating'], epochs=10, batch_size=64)


def evaluate(model, test_data):
    # Evaluate the model and print the loss and accuracy values
    loss, accuracy = model.evaluate([test_data['userId'], test_data['movieId']], test_data['rating'])
    print(f"Loss: {loss}, accuracy: {accuracy}")


def recommend(test_data, user_mapping, reverse_user_mapping, movie_mapping, movies_data, model):
    user_id = user_mapping[USERID]

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

    return recommendations


def print_recommendations_for_user(recommendations, reverse_movie_mapping):
    print(f"Recommendations for you: ")
    for _, movie in recommendations.head(5).iterrows():
        movie_id = movie['movieId']
        if movie_id in reverse_movie_mapping:
            print(
                f"Movie ID: {reverse_movie_mapping[movie_id]} - Title: {movie['title']} - Genres: {movie['genres']} - Predicted Rating: {movie['predicted_rating']}")
        else:
            print(
                f"Movie ID: {movie_id} - Title: {movie['title']} - Genres: {movie['genres']} - Predicted Rating: {movie['predicted_rating']} (Movie ID not found in mapping)")


def define_model(num_users, num_movies, embedding_dim):
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
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


def five_recommendations_for_user():
    ratings_data, movies_data = load_data()
    ratings_data = prompt_user(movies_data, ratings_data)
    ratings_data, user_mapping, reverse_user_mapping, movie_mapping, reverse_movie_mapping = preprocess_data(
        ratings_data)

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

    model = define_model(num_users, num_movies, embedding_dim)

    train(model, train_data)
    # evaluate(model, test_data)
    recommendations = recommend(test_data, user_mapping, reverse_user_mapping, movie_mapping, movies_data, model)
    print_recommendations_for_user(recommendations, reverse_movie_mapping)


def main():
    # menu for user to choose which function to run
    print("Welcome to the Movie Recommender System!")
    print("Please choose one of the following options:")
    print("1. Recommend 5 movies for a user")

    choice = input("Enter your choice: ")
    if choice == "1":
        five_recommendations_for_user()

    return 0


if __name__ == "__main__":
    main()