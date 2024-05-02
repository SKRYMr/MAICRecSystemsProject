import numpy as np
import pandas as pd
import os
import pickle
import sys
from typing import Dict

# Suppress the warning on chained assignments which doesn't apply to our case
pd.options.mode.chained_assignment = None  # default='warn'

RATINGS_DAT_FILE = "./data/ratings.dat"
MOVIES_DAT_FILE = "./data/movies.dat"
USERS_DAT_FILE = "./data/users.dat"
DB_PICKLE_PATH = "./data/database.pickle"

MIN_USER_RATING = 4     # Movies with rating equal or higher than this number are used to create the user profile.
MIN_GENRE_OVERLAP = 0.2 # Movies with lower overlaps than this will not be considered.
N_GENRES = 5            # Number of genres to consider in genre count recommendation.

class MovieLens:
    def __init__(self):
        try:
            self.ratings = pd.read_table(RATINGS_DAT_FILE, engine="python", sep="::", usecols=[0, 1, 2], names=["user_id", "movie_id", "rating"], dtype={"rating": np.float32})

            self.movies = pd.read_table(MOVIES_DAT_FILE, engine="python", sep="::", names=["movie_id", "title", "genres"], encoding="latin")
            self.movies["genres"] = self.movies["genres"].apply(lambda x: x.split("|"))  # convert genres to list

            self.users = set(pd.read_table(USERS_DAT_FILE, engine="python", sep="::", usecols=[0], names=["user_id"])["user_id"].unique())

        except FileNotFoundError as err:
            print("Some database files are missing.")
            sys.exit(0)

    def get_user_profile(self, user_id: int, normalized=False):
        user_ratings = self.ratings[(self.ratings["user_id"] == user_id) & (self.ratings["rating"] >= MIN_USER_RATING)]
        user_movies = self.movies[self.movies["movie_id"].isin(user_ratings["movie_id"])]

        user_genres_weighted = user_movies["genres"].explode().value_counts(normalize=normalized).to_dict()

        return user_genres_weighted, pd.merge(user_movies, user_ratings, on="movie_id", how="inner").drop(columns="user_id")

def check_user_id(user_id: int, users: set) -> bool:
    if user_id not in users:
        print("User not found, try again")
        return False
    return True

def show_user_profile(genres: Dict[str, int], movies: pd.DataFrame, n: int = 10):
    print("\nUser Profile =======================")

    print("\nBest movies")
    for i, (_, row) in enumerate(movies.sort_values(by="rating", ascending=False).iterrows()):
        print(f"{i+1}) {row['title']} - {', '.join(row['genres'])} - {row['rating']}")
        if (i+1) >= n:
            break

    print(f"\nGenre")
    for k, v in genres.items():
        print(f"{k} - {v}")

    print(f"\n===================================")

def keyword_similarity(user_genres: Dict[str, int], movies: pd.DataFrame):
    genres = set(user_genres.keys())
    n_user_genres = len(genres)
    movies["keyword_similarity"] = movies["genres"].apply(lambda x: (2 * len(set(x).intersection(genres))) / (n_user_genres + len(x)))
    return movies

# Remove movies with overlap below the given threshold MIN_GENRE_OVERLAP. Calculate "popularity" based on number of ratings.
def movie_popularity(movies: pd.DataFrame, ratings: pd.DataFrame):
    movies = movies[movies['keyword_similarity'] > MIN_GENRE_OVERLAP]
    
    movie_counts = ratings['movie_id'].value_counts().reset_index()
    movie_counts.columns = ['movie_id', 'num_ratings']
    
    return pd.merge(movies, movie_counts, on='movie_id', how='left').fillna(0)

def movie_popularity_genres(movies, user_genres):
    user_genres = pd.Series(user_genres)

    # Get genre-count weights
    user_genres = user_genres/user_genres.sum()

    # Keep only the user's N most favourite genres
    # If we kept all genres the genre overlap could have bigger effect than weights
    user_genres = user_genres[:N_GENRES]

    # Normalize movie popularity
    movies.num_ratings = movies.num_ratings / movies.num_ratings.max()

    movies["gc_similarity"] = movies["genres"].apply(lambda x: user_genres.loc[list(set(x).intersection(user_genres.index))].sum())
    movies["gc_similarity"] = movies["gc_similarity"] * movies["num_ratings"]

    return movies

def show_recommendations(movies: pd.DataFrame, column: str, title:str="Keyword", n: int = 10):
    print(f"\n{title} Recommendations ===================================\n")
    for i, (_, row) in enumerate(movies.sort_values(column, ascending=False).iterrows()):
        print(f"{i+1}) {row['title']} - {', '.join(row['genres'])}")
        print(f"Keyword Similarity: {row['keyword_similarity']:.2f}")
        if column != "keyword_similarity":
            print(f"{' '.join(column.split('_')).title()}: {row[column]:.2f}")
        print()
        if (i+1) >= n:
            break
    print(f"===================================")

if __name__ == "__main__":
    if os.path.isfile(DB_PICKLE_PATH):
        with open(DB_PICKLE_PATH, "rb") as fin:
            database = pickle.load(fin)
    else:
        database = MovieLens()
        with open(DB_PICKLE_PATH, "wb") as fout:
            pickle.dump(database, fout)

    while True:
        try:
            user_id = int(input("Enter user id: "))
            if check_user_id(user_id, database.users):
                break
        except ValueError as err:
            print("User ID must be a number.")

    user_genres, user_movies = database.get_user_profile(user_id, False)
    show_user_profile(user_genres, user_movies)

    # We recommend only movies that the user has not rated
    relevant_movies = database.movies[~database.movies["movie_id"].isin(user_movies["movie_id"])]

    # Recommend movies that have the greatest overlap in genres with the set of the user's favourite genres.
    # Clearly with this strategy if the user has liked many different movies the spread of genres is such
    # that the best recommended movies will simply be the ones that have the most genres assigned to them.
    movies = keyword_similarity(user_genres, relevant_movies)
    show_recommendations(movies, "keyword_similarity")
    
    # Will recommend based on movies that have keyword_similarity > 0. 
    # And then with number of user ratings on those movies.
    # Could possibly be improved alot by introducing some kind of weight. Now we are considering all ratings.
    # With a low overlap-threshold it will most of the times give the same movie recommendation.
    movies_with_overlap = movie_popularity(movies, database.ratings)
    show_recommendations(movies_with_overlap, "num_ratings", title="Movie Popularity")

    # Considers genre overlap, genre count of user profile and number of ratings
    # Top n genres are considered and weighted based on rating percentage
    movies_genre_count = movie_popularity_genres(movies_with_overlap, user_genres)
    show_recommendations(movies_genre_count, "gc_similarity", title="Genre-Count")
