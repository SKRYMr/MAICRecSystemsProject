import json
import os
import requests

from db import add_to_googledrivemovies

ROOT = "extracted_content_ml-latest"
API_KEY = '347f057cc85997cb119a516c59c66063'


def get_poster(poster, tmdb_id):
    if poster is not None:
        return poster

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&append_to_response=images"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()

    if response.status_code == 200:
        return data["poster_path"]

    return None


def get_array_items(items):
    if items is None:
        return ["N/A"]
    elif len(items) == 0:
        return ["N/A"]
    elif len(items[0]) == 0:
        return ["N/A"]

    return items


def get_str_item(item):
    if item == "":
        return "N/A"
    elif item is None:
        return "N/A"

    return item


def get_num_item(item):
    try:
        if item is None:
            return -1

        return float(item)
    except:
        return -1


def get_release_year(year):
    try:
        return int(year)
    except:
        return -1


def extract_data_from_file(data):
    movie_id = data["movielensId"]  # int
    tmdb_id = data["movielens"]["tmdbMovieId"]  # int
    imdb_link = get_str_item(data["imdb"]["imdbLink"])  # str
    languages = get_array_items(data["movielens"]["languages"])  # List[str]
    release_date = get_str_item(data["movielens"]["releaseDate"])  # str
    release_year = get_release_year(data["movielens"]["releaseYear"])  # str
    runtime = get_num_item(data["movielens"]["runtime"])  # int
    youtube_trailer_video_ids = data["movielens"]["youtubeTrailerIds"]  # List[str]
    synopsis = get_str_item(data["movielens"]["plotSummary"])  # str
    num_ratings = get_num_item(data["movielens"]["numRatings"])  # int
    avg_ratings = get_num_item(data["movielens"]["avgRating"])  # int
    poster = get_poster(data["movielens"]["posterPath"], data["movielens"]["tmdbMovieId"])  # str
    title = get_str_item(data["movielens"]["originalTitle"])  # str
    genres = get_array_items(data["movielens"]["genres"])  # List[str]
    actors = get_array_items(data["movielens"]["actors"])  # List[str]
    directors = get_array_items(data["movielens"]["directors"])  # List[str]
    writers = get_array_items(data["imdb"]["writers"])  # List[str]
    age_ratings = get_str_item(data["movielens"]["mpaa"])  # str

    try:
        tmdb_recommendations = data["tmdb"]["recommendations"]  # List[str], rec based on tmdb_id
    except:
        tmdb_recommendations = []

    return [
        movie_id,
        tmdb_id,
        title,
        genres,
        age_ratings,
        avg_ratings,
        num_ratings,
        languages,
        actors,
        directors,
        writers,
        release_year,
        release_date,
        runtime,
        imdb_link,
        poster,
        youtube_trailer_video_ids,
        synopsis,
        tmdb_recommendations
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
        "languages",
        "actors",
        "directors",
        "writers",
        "release_year",
        "release_date",
        "runtime",
        "imdb_link",
        "poster",
        "youtube_trailer_video_ids",
        "synopsis",
        "tmdb_recommendations"
    ]

    for file_name in os.listdir(ROOT):
        with open(os.path.join(ROOT, file_name), "r") as file:
            data = extract_data_from_file(json.load(file))

            # Convert all list to str
            item = []
            for d in data:
                if isinstance(d, list):
                    item.append("; ".join([str(e) for e in d]).replace("é", "e"))
                elif isinstance(d, str):
                    item.append(d.replace("é", "e"))
                else:
                    item.append(d)

            add_to_googledrivemovies(item)

