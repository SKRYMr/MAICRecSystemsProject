import json
import os
import pandas as pd
from tqdm import tqdm

ROOT = "extracted_content_ml-latest"


def extract_data_from_file(data):
    movie_id = data["movielensId"]  # int
    tmdb_id = data["movielens"]["tmdbMovieId"]  # int
    imdb_link = data["imdb"]["imdbLink"]  # str

    language = data["movielens"]["languages"]  # List[str]
    release_date = data["movielens"]["releaseDate"]  # str
    release_year = data["movielens"]["releaseYear"]  # str
    runtime = data["movielens"]["runtime"]  # int
    youtube_trailer_video_id = data["movielens"]["youtubeTrailerIds"]  # List[str]
    summary = data["movielens"]["plotSummary"]  # str
    num_ratings = data["movielens"]["numRatings"]  # int
    avg_ratings = data["movielens"]["avgRating"]  # int
    poster = data["movielens"]["posterPath"]  # str
    title = data["movielens"]["originalTitle"]  # str
    genres = data["movielens"]["genres"]  # List[str]
    actors = data["movielens"]["actors"]  # List[str]
    directors = data["movielens"]["directors"]  # List[str]
    writer = data["imdb"]["writers"]  # List[str]
    age_ratings = data["movielens"]["mpaa"]  # str

    try:
        tmdb_recommendation = data["tmdb"]["recommendations"]  # List[str], rec based on tmdb_id
    except:
        tmdb_recommendation = ""

    return [
        movie_id,
        tmdb_id,
        title,
        genres,
        age_ratings,
        avg_ratings,
        num_ratings,
        language,
        actors,
        directors,
        writer,
        release_year,
        release_date,
        runtime,
        imdb_link,
        poster,
        youtube_trailer_video_id,
        summary,
        tmdb_recommendation
    ]


if __name__ == "__main__":

    COL_NAMES = [
        "movie_id",
        "tmdb_id",
        "title",
        "genres",
        "age_ratings",
        "avg_ratings",
        "num_ratings",
        "language",
        "actors",
        "directors",
        "writer",
        "release_year",
        "release_date",
        "runtime",
        "imdb_link",
        "poster",
        "youtube_trailer_video_id",
        "summary",
        "tmdb_recommendation"
    ]

    data = []

    for file_name in tqdm(os.listdir(ROOT)):
        with open(os.path.join(ROOT, file_name), "r") as file:
            data.append(extract_data_from_file(json.load(file)))

    df = pd.DataFrame(data, columns=COL_NAMES).to_csv(f"{ROOT}_data.csv")
