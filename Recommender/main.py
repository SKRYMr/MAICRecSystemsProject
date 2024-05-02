import numpy as np
import pandas as pd
import os
import pickle
import random
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Set, Literal
import time

from .ex3 import find_k_nearest, get_top_movies, TIME_FUNCS, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"
DB_PICKLE_PATH = "/Recommender/data/database.pickle"

POSITIVE_RATING_THRESHOLD = 3

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def get_movies_recommendations(user_id: int, users: Set[int], ratings: pd.DataFrame, movies: pd.DataFrame, minimum_ratings: int = 5, metric: Literal["neighbours_average", "bias_correction"] = "neighbours_average"):
    neighbours, similarities = find_k_nearest(user_id, users, ratings, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS)
    return get_top_movies(user_id, neighbours, ratings, movies, similarities, minimum_ratings, metric)

class MovieLens:
    def __init__(self):
        try:
            self.ratings = pd.read_table(RATINGS_DAT_FILE, engine="python", sep="::", usecols=[0, 1, 2],
                                         names=["user_id", "movie_id", "rating"], dtype={"rating": np.float32})

            self.movies = pd.read_table(MOVIES_DAT_FILE, engine="python", sep="::",
                                        names=["movie_id", "title", "genres"], encoding="latin")
            self.movies["genres"] = self.movies["genres"].apply(lambda x: x.split("|"))  # convert genres to list

            self.users = set(pd.read_table(USERS_DAT_FILE, engine="python", sep="::", usecols=[0], names=["user_id"])[
                                 "user_id"].unique())
        except FileNotFoundError:
            print("Some database files are missing.")
            sys.exit(0)

    def use_subset(self, size=0.5):
        # Size = what percentage of orignal data to use
        self.movies, _ = train_test_split(self.movies, train_size=size, random_state=RANDOM_SEED)
        self.users, _ = train_test_split(list(self.users), train_size=size, random_state=RANDOM_SEED)
        self.users = set(self.users)

        # Remove ratings of users and movies which are not in the subset
        self.ratings = self.ratings[
            (self.ratings["user_id"].isin(self.users)) & (self.ratings["movie_id"].isin(self.movies["movie_id"]))]

    def split_ratings(self, test_size=0.2):
        return train_test_split(self.ratings, test_size=test_size, random_state=RANDOM_SEED)
    
def get_predictions(test_ratings: pd.DataFrame, train_ratings: pd.DataFrame, users: set, movies: pd.DataFrame, num_users: int = 10, fillNaValue: Literal["zero", "user_mean_rating"] = "zero") -> pd.DataFrame:
    predictions = []
    
    for i, user_id in enumerate(tqdm(test_ratings["user_id"].unique(), desc="Computing predictions...", unit="user", total=num_users if num_users > -1 else None)):
        s = time.time() if TIME_FUNCS else 0
        user_test_movies = test_ratings[test_ratings["user_id"] == user_id]
        recommended_movies = get_movies_recommendations(user_id, users, train_ratings, movies)

        df = pd.merge(user_test_movies, recommended_movies, on="movie_id", how="left") # merge on movie_id which are test set
        if fillNaValue == "zero":
            df["rating_y"] = df["rating_y"].fillna(0) # if the pred rating does not exist, use 0
        elif fillNaValue == "user_mean_rating":
            user_mean_rating = train_ratings[train_ratings['user_id'] == user_id]['rating'].mean()
            df["rating_y"] = df["rating_y"].fillna(user_mean_rating) # if the pred rating does not exist, use user average rating

        df["rating_y"] = df["rating_y"].apply(round)
        predictions += df[["rating_x", "rating_y"]].values.tolist()
        if TIME_FUNCS: print(f"One iteration time: {time.time() - s:.3f}")
        if i == num_users:
            break
    return predictions

def get_user_predictions(test_ratings: pd.DataFrame, train_ratings: pd.DataFrame, users: set, movies: pd.DataFrame, num_users: int = 10, fillNaValue: Literal["zero", "user_mean_rating"] = "zero", minimum_ratings: int = 5, metric: Literal["neighbours_average", "bias_correction"] = "neighbours_average") -> pd.DataFrame:
    user_predictions = {}
    
    for i, user_id in enumerate(tqdm(train_ratings["user_id"].unique(), desc="Computing user predictions...", unit="user", total=num_users if num_users > -1 else None)):
        s = time.time() if TIME_FUNCS else 0
        user_test_ratings = test_ratings[test_ratings["user_id"] == user_id]
        recommended_movies = get_movies_recommendations(user_id, users, train_ratings, movies, minimum_ratings, metric=metric)

        df = pd.merge(user_test_ratings, recommended_movies, on="movie_id", how="left") # merge on movie_id which are test set
        if fillNaValue == "zero":
            df["rating_y"] = df["rating_y"].fillna(0) # if the pred rating does not exist, use 0
        elif fillNaValue == "user_mean_rating":
            user_mean_rating = train_ratings[train_ratings['user_id'] == user_id]['rating'].mean()
            df["rating_y"] = df["rating_y"].fillna(user_mean_rating) # if the pred rating does not exist, use user average rating

        df["rating_y"] = df["rating_y"].apply(round)
        predictions = df[["movie_id", "rating_x", "rating_y"]].values.tolist()
        predictions.sort(key=lambda x: x[-1], reverse=True)
        user_predictions[user_id] = predictions[:10]
        if TIME_FUNCS: print(f"One iteration time: {time.time() - s:.3f}")
        if i == num_users:
            break
    return user_predictions


def calculate_error(predictions):
    # Predictions[:, 1] = rating_y = the prediction. Column 0 = rating_x = the rating from test set
    MAE = np.mean(np.abs(np.array(predictions)[:, 1] - np.array(predictions)[:, 0]))
    RMSE = np.sqrt(np.mean((np.array(predictions)[:, 1] - np.array(predictions)[:, 0]) ** 2))
    return MAE, RMSE 

def print_error(mae, rmse, rows_covered):
    print(f"Rows covered: {rows_covered}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

def compute_scores(user_predictions: pd.DataFrame, threshold: int = 3):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for preds in user_predictions.values():
        for _, target, pred in preds:
            if pred > threshold and target > threshold:
                true_positives += 1
            elif pred <= threshold and target > threshold:
                false_negatives += 1
            elif pred > threshold and target <= threshold:
                false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2*(precision*recall)/(precision+recall)

    return recall, precision, f1

if __name__ == "__main__":
    if os.path.isfile(DB_PICKLE_PATH):
        with open(DB_PICKLE_PATH, "rb") as f:
            database = pickle.load(f)
    else:
        database = MovieLens()
        with open(DB_PICKLE_PATH, "wb") as f:
            pickle.dump(database, f)

    database.use_subset(0.5)

    train_ratings, test_ratings = database.split_ratings()

    ### Task 1

    # Predictions when filling "non-recommendations" with 0 rating.
    s = time.time() if TIME_FUNCS else 0
    predictions = get_predictions(test_ratings, train_ratings, database.users, database.movies, fillNaValue="zero")
    if TIME_FUNCS: print(f"Time to get predictions: {time.time() - s}")
    MAE, RMSE = calculate_error(predictions)
    print_error(MAE, RMSE, len(predictions))
    
    # Predictions when filling "non-recommendations" with the users average rating.
    print("---------------------")
    s = time.time() if TIME_FUNCS else 0
    predictions = get_predictions(test_ratings, train_ratings, database.users, database.movies, fillNaValue="user_mean_rating")
    if TIME_FUNCS: print(f"Time to get predictions: {time.time() - s}")
    MAE, RMSE = calculate_error(predictions)
    print_error(MAE, RMSE, len(predictions))

    ### Task 2

    # Compute predictions for each user
    user_predictions = get_user_predictions(test_ratings, train_ratings, database.users, database.movies, num_users=1000, fillNaValue="user_mean_rating", minimum_ratings=3, metric="neighbours_average")
    for user, preds in user_predictions.items():
        print("-"*45)
        print(f"USER {user}")
        for movie in preds:
            print(f"Movie: {database.movies[database.movies['movie_id'] == int(movie[0])].get('title').to_string(name=False, dtype=False, index=False)}")
            print(f"True:\t{movie[1]}")
            print(f"Pred:\t{movie[2]}")

    # Compute precision, recall and F1
    precision, recall, f1 = compute_scores(user_predictions, threshold=POSITIVE_RATING_THRESHOLD)

    print("-"*45)
    print(f"{'PRECISION:':>10} {precision:.4f}")
    print(f"{'RECALL:':>10} {recall:.4f}")
    print(f"{'F1:':>10} {f1:.4f}")

    ### RESULTS:
    """
    All test performed with fixed random seed (42)

    TEST 1:
    -- SAMPLE SIZE: 1000 users
    -- AVERAGE EXECUTION TIME: 1.8 seconds per user
    -- PRECISION: 0.8250
    -- RECALL: 0.7235
    -- F1: 0.7709
    - fillNaValue = "zero"
    - minimum_ratings = 3
    - NEIGHBOURHOOD_SIZE = 50
    - POSITIVE_RATING_THRESHOLD = 3
    - MINIMUM_COMMON_SIZE = 0
    - Prediction method: Neighbours average rating

    TEST 2:
    -- SAMPLE SIZE: 1000 users
    -- AVERAGE EXECUTION TIME: 1.8 seconds per user
    -- PRECISION: 0.9300
    -- RECALL: 0.6798
    -- F1: 0.7854
    - fillNaValue = "user_mean_rating"
    - minimum_ratings = 3
    - NEIGHBOURHOOD_SIZE = 50
    - POSITIVE_RATING_THRESHOLD = 3
    - MINIMUM_COMMON_SIZE = 0
    - Prediction method: Neighbours average rating

    TEST 3:
    -- SAMPLE SIZE: 200 users
    -- AVERAGE EXECUTION TIME: 16 seconds per user
    -- PRECISION: 0.7271
    -- RECALL: 0.6944
    -- F1: 0.7104
    - fillNaValue = "user_mean_rating"
    - minimum_ratings = 3
    - NEIGHBOURHOOD_SIZE = 50
    - POSITIVE_RATING_THRESHOLD = 3
    - MINIMUM_COMMON_SIZE = 0
    - Prediction method: Average User Rating + Bias Correction

    TEST 4:
    -- SAMPLE SIZE: 1000 users
    -- AVERAGE EXECUTION TIME: 1.4 seconds per user
    -- PRECISION: 0.9300
    -- RECALL: 0.6798
    -- F1: 0.7854
    - fillNaValue = "user_mean_rating"
    - minimum_ratings = 3
    - NEIGHBOURHOOD_SIZE = 50
    - POSITIVE_RATING_THRESHOLD = 3
    - MINIMUM_COMMON_SIZE = 5
    - Prediction method: Neighbours average rating
    """
    
    
    
  
    
