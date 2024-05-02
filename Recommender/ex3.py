import numpy as np
import os
import pandas as pd
import pickle
import sys
import time
from typing import Tuple, Literal

TIME_FUNCS = False
NEIGHBOURHOOD_SIZE = 50
MINIMUM_COMMON_RATINGS = 5

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"
DB_PICKLE_PATH = "./data/database.pickle"

class MovieLens():
    def __init__(self):
        try:
            self.ratings = pd.read_table(RATINGS_DAT_FILE, engine="python", sep="::", usecols=[0, 1, 2],
                                         names=["user_id", "movie_id", "rating"], dtype={"rating": np.float32})
            self.movies = pd.read_table(MOVIES_DAT_FILE, engine="python", sep="::",
                                        names=["movie_id", "title", "genres"], encoding="latin")
            self.users = set(pd.read_table(USERS_DAT_FILE, engine="python", sep="::", usecols=[0], names=["user_id"])[
                                 "user_id"].unique())
            self.similarities = {}
        except FileNotFoundError as err:
            print("Some database files are missing.")
            sys.exit(0)

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        return user_ratings

    def get_user_movies(self, user_id: int) -> pd.DataFrame:
        return pd.merge(self.get_user_ratings(user_id), database.movies, on="movie_id")

    def get_movie_info(self, movie_id: int) -> pd.DataFrame:
        movie_info = self.movie_data[self.movie_data["movie_id"] == movie_id]
        return movie_info

    def get_user_mean(self, user_id):
        user_ratings = self.get_user_ratings(user_id)['rating']
        if user_ratings.empty:
            return None
        return user_ratings.mean()


def show_movies(movies: pd.DataFrame, n_movies: int = 15, recommendation: bool = False):
    if n_movies > 15:
        n_movies = 15

    if recommendation:
        print(f"\n{n_movies if len(movies) > n_movies else len(movies)} Best Movies For User =========================")
    else:
        print(f"\n{n_movies if len(movies) > n_movies else len(movies)} Movies Rated By User =========================")
    for i, row in movies.iterrows():
        genres = row["genres"].replace("|", ", ") if type(row["genres"]) != list else row["genres"]
        print(f"{i + 1}) {row['title']} - {genres} - {row['rating']:.3f}")
        n_movies -= 1
        if n_movies == 0:
            break
    print(f"================================================\n")


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


def get_top_movies(user_id: int, neighbours: list, ratings: pd.DataFrame, movies: pd.DataFrame, similarities: pd.DataFrame = {}, minimum_ratings: int = 5, metric: Literal["neighbours_average", "bias_correction"] = "neighbours_average") -> pd.DataFrame:
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


if __name__ == "__main__":
    if os.path.isfile(DB_PICKLE_PATH):
        with open(DB_PICKLE_PATH, "rb") as f:
            database = pickle.load(f)
    else:
        database = MovieLens()
        with open(DB_PICKLE_PATH, "wb") as f:
            pickle.dump(database, f)
    while True:
        try:
            user_id = int(input("User id: "))
        except ValueError as err:
            print("User ID must be a number.")
        if check_user_id(user_id, database.users):
            break

    show_movies(database.get_user_movies(user_id).sort_values("rating", ascending=False, ignore_index=True), n_movies=15)
    neighbours, similarities = find_k_nearest(user_id, database.users, database.ratings, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS)
    get_top_movies(user_id, neighbours, database.ratings, database.movies, similarities)
