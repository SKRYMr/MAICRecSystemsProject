import numpy as np
import math
import pandas as pd
import sys
import time
from collections.abc import Iterable
from typing import Tuple, Literal, Union

TIME_FUNCS = False
NEIGHBOURHOOD_SIZE = 50
MINIMUM_COMMON_RATINGS = 5
# How many highest-value ratings (i.e. how many users) we need
# to get recommendations for a movie based on users' average ratings.
# See views.recommend_neighbours
BEST_STAR_RATINGS = 5
# What percentage of target movie's best raters must have rated another movie
# for that movie to be included in recommendations.
# See views.recommend_neighbours
MINIMUM_RATINGS_PERCENT = 0.15
# Parameter to make bigger queries compatible with lower versions of sqlite3.
MAX_IDS_PER_EXCLUSION = 1000
# Parameter to penalize excessively popular items appearing too frequently
# or being scored too high. This is effectively multiplied with a number between
# 0 and (POPULARITY_CENTRE - popularity)² where popularity is a number between 0 and 1
# and then subtracted from the predicted rating for the movie.
# The range of popularities is HEAVILY skewed towards 0.
# 71.3% of movies have a popularity between 0 and 0.05%
# Here is a short summary of penalties applied to movies of the corresponding popularity:
# POPULARITY: PENALTY
# 1e-05: 0.0
# 5e-05: 0.01431
# 0.0001: 0.02561
# 0.0005: 0.06791
# 0.001: 0.09801
# 0.005: 0.22271
# 0.01: 0.3156
# 0.05: 0.70682
# 0.1: 0.9998
# 0.2: 1.41407
# 0.3: 1.73194
# 0.4: 1.9999
# 0.5: 2.23598
# 0.6: 2.44941
# 0.7: 2.64568
# 0.8: 2.82836
# 0.9: 2.99993
# 1.0: 3.16221
POPULARITY_PENALTY_WEIGHT = 100000000
# Parameter to define the centre around which the popularity penalty is distributed.
# The idea is to penalize both extremely popular movies and absolutely unknown ones.
# Since the popularity is expressed as a decimal percentage, ideally a centre of 0.5 would
# imply perfect symmetry where the maximum penalty is applied equally to very popular
# (popularity close to 1) and very niche (popularity close to 0) movies. A lower value
# will penalize niche movies less and a higher value will penalize them more.
# However, due to how ratings are distributed in the database, the actual popularity is
# heavily skewed towards the zero.
POPULARITY_PENALTY_CENTRE = 0.00001

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"
DB_PICKLE_PATH = "./data/database.pickle"


class MovieLens:
    def __init__(self):
        try:
            self.ratings = pd.read_table(RATINGS_DAT_FILE, engine="python", sep="::", usecols=[0, 1, 2],
                                         names=["user_id", "movie_id", "rating"], dtype={"rating": np.float32})
            self.movies = pd.read_table(MOVIES_DAT_FILE, engine="python", sep="::",
                                        names=["movie_id", "title", "genres"], encoding="latin")
            self.users = set(pd.read_table(USERS_DAT_FILE, engine="python", sep="::", usecols=[0], names=["user_id"])[
                                 "user_id"].unique())
            self.similarities = {}
        except FileNotFoundError:
            print("Some database files are missing.")
            sys.exit(0)

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        return user_ratings

    def get_user_movies(self, user_id: int) -> pd.DataFrame:
        return pd.merge(self.get_user_ratings(user_id), self.movies, on="movie_id")

    def get_movie_info(self, movie_id: int) -> pd.DataFrame:
        movie_info = self.movie_data[self.movie_data["movie_id"] == movie_id]
        return movie_info

    def get_user_mean(self, user_id):
        user_ratings = self.get_user_ratings(user_id)['rating']
        if user_ratings.empty:
            return None
        return user_ratings.mean()


def check_user_id(user_id: int, users: set) -> bool:
    if user_id not in users:
        print("User not found.")
        print(f"Minimum user ID: {min(users)}")
        print(f"Maximum user ID: {max(users)}")
        return False
    return True


def pearson_correlation(col1, col2) -> float:
    if len(col1) == 0 or len(col1) < 5:
        return 0
    col1_mean = col1.mean()
    col2_mean = col2.mean()
    a = ((col1 - col1_mean) * (col2 - col2_mean)).sum()
    b = ((col1 - col1_mean) ** 2).sum()
    c = ((col2 - col2_mean) ** 2).sum()
    if b == 0 or c == 0:
        b = b + 0.02
        c = c + 0.02
    return a / ((b * c) ** 0.5)


def find_k_nearest(user_id: int, users: set, ratings: pd.DataFrame, k: int, minimum_common_ratings: int = 5) -> Tuple[list, dict]:
    s = time.time() if TIME_FUNCS else 0
    similarities = {}
    user_ratings = ratings[ratings.user_id == int(user_id)]
    ratings_comparison = pd.merge(user_ratings, ratings, on="movie_id")
    for id in users - {user_id}:
        other_user_ratings = ratings_comparison[ratings_comparison.user_id_y == int(id)]
        if len(other_user_ratings) < minimum_common_ratings:
            continue
        similarities[id] = pearson_correlation(other_user_ratings["rating_x"], other_user_ratings["rating_y"])
    similarities_sorted = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    users = [user[0] for user in similarities_sorted[:k]]
    if TIME_FUNCS: print(f"Time to get nearest neighbours: {time.time() - s:.3f}")
    return users, similarities


def get_top_movies(user_id: int, neighbours: list, ratings: pd.DataFrame, movies: pd.DataFrame, similarities: dict = {},
                   minimum_ratings: int = 5,
                   metric: Literal["neighbours_average", "bias_correction"] = "neighbours_average") -> pd.DataFrame:
    s = time.time() if TIME_FUNCS else 0
    user_ratings = ratings[ratings["user_id"] == user_id]
    neighbours_ratings = ratings[ratings.user_id.isin(neighbours)]
    if metric == "bias_correction" and len(similarities) > 0:
        user_avg_rating = user_ratings.rating.mean()
        avg_neighbours_ratings = neighbours_ratings.groupby("user_id").rating.mean().reset_index()
        similarities_sum = sum(similarities.values())
        predicted_ratings = [(movie_id, user_avg_rating +
                                       (sum([similarities[neighbour_id]*(rating.item() - avg_neighbours_ratings[avg_neighbours_ratings["user_id"] == neighbour_id].rating.item() if not (rating := ratings[(ratings["user_id"] == neighbour_id) & (ratings["movie_id"] == movie_id)].rating).empty else 0) for neighbour_id in neighbours])
                                         / similarities_sum))
                              for movie_id in movies[movies.movie_id.isin(neighbours_ratings.movie_id.unique())].movie_id.unique()]
        predicted_ratings = pd.DataFrame(predicted_ratings, columns=["movie_id", "rating"])
        predicted_ratings.sort_values("rating", ascending=False, inplace=True)
    else:
        neighbours_ratings = neighbours_ratings[~neighbours_ratings.movie_id.isin(user_ratings.movie_id)]
        neighbours_ratings = neighbours_ratings[neighbours_ratings["movie_id"].map(neighbours_ratings["movie_id"].value_counts()) >= minimum_ratings]
        neighbours_ratings = neighbours_ratings.groupby("movie_id").rating.mean().reset_index()
        predicted_ratings = neighbours_ratings.sort_values("rating", ascending=False)
    if TIME_FUNCS: print(f"Time to get top movies: {time.time() - s:.3f}")
    return pd.merge(predicted_ratings, movies, on="movie_id")


def get_user_item_table(user_id: int, df_ratings: pd.DataFrame) -> pd.DataFrame:
    user_item_table = df_ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0)
    user_vector = user_item_table.iloc[user_id - 1]
    user_item_table.drop(index=user_id, inplace=True)

    # Calculate user similarity
    user_item_table["score"] = user_item_table.apply(lambda x: user_vector.corr(x), axis=1)

    return user_item_table


def get_best_users(user_item_table: pd.DataFrame, min_corr: float = 0.25) -> pd.DataFrame:
    best_users = user_item_table[user_item_table["score"] > min_corr]

    if best_users.empty:
        print("No users found, reduce min correlation")
        sys.exit(0)
    else:
        print(f"Computing recommendations on {len(best_users)} users...\n")

    return best_users


def rate_movies_by_best_users(best_users: pd.DataFrame, user_movies: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(best_users.apply(np.average, axis=0), columns=["avg_rating"]).sort_values("avg_rating", ascending=False).drop(user_movies["movie_id"])


def compute_popularity_penalty(popularity: Union[float, Iterable], index: Iterable = None):
    weight = lambda x: POPULARITY_PENALTY_WEIGHT / math.sqrt((x / POPULARITY_PENALTY_CENTRE) ** 3)
    penalty = lambda x: weight(x) * (x - POPULARITY_PENALTY_CENTRE) ** 2
    return popularity.__class__([penalty(pop) for pop in popularity], index) \
        if isinstance(popularity, Iterable) \
        else penalty(popularity)
