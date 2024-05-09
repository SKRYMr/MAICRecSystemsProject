import numpy as np
import pandas as pd
import random
from typing import Set, Literal

from .core import find_k_nearest, get_top_movies, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS
from .parse import preprocess_pipeline, clean_pipeline, ft
from RecSystems5.settings import DEBUG

POSITIVE_RATING_THRESHOLD = 3

if DEBUG:
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def get_movies_recommendations(user_id: int, users: Set[int], ratings: pd.DataFrame, movies: pd.DataFrame,
                               minimum_ratings: int = 5,
                               metric: Literal["neighbours_average", "bias_correction"] = "neighbours_average"):
    neighbours, similarities = find_k_nearest(user_id, users, ratings, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS)
    return get_top_movies(user_id, neighbours, ratings, movies, similarities, minimum_ratings, metric)


def compute_synopsis_vec(text: str) -> np.array:
    preprocessed_text = preprocess_pipeline(clean_pipeline(text))
    vec = [ft[word] for word in preprocessed_text.split()]
    vec = np.mean(vec, axis=0)
    return vec


def format_movie_recommendations(recommendations: pd.DataFrame, top_n: int = 0) -> pd.DataFrame:
    recommendations["actors"] = recommendations["actors"].apply(
        lambda x: x.replace("[", "").replace("(", "").replace("'", "").replace("]", "").replace(")", "").split(", "))
    recommendations["genres"] = recommendations["genres"].apply(
        lambda x: x.replace("[", "").replace("]", "").replace("'", "").split(", "))
    recommendations["poster"] = recommendations["poster"].apply(lambda x: "https://image.tmdb.org/t/p/original" + x)
    recommendations["rating"] = recommendations["rating"].apply(lambda x: round(x, 1))
    return recommendations.head(top_n) if top_n > 0 else recommendations
