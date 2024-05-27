import json
import os
import requests
from datetime import datetime
from typing import List, Dict, Tuple
from .models import Movie

GOOGLE_DRIVE_ROOT = os.path.join(os.getcwd(), "extracted_content_ml-latest")
API_KEY = "347f057cc85997cb119a516c59c66063"
DATETIME_FORMAT = "%Y-%m-%d"


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
        return None
    elif len(items) == 0:
        return None
    elif len(items[0]) == 0:
        return None

    return items


def get_str_item(item):
    if item == "":
        return None
    elif item is None:
        return None

    return item


def get_num_item(item):
    try:
        if item is None:
            return None

        return float(item)
    except Exception as e:
        print("Exception encountered in get_num_item:")
        print(e)
        return None


def get_release_year(year):
    try:
        return int(year)
    except ValueError:
        return None
    except Exception as e:
        print("Unexpected exception encountered in get_release_year:")
        print(e)
        return None


def get_keywords(keywords_list: List[Dict]):
    if not keywords_list:
        return None
    keywords = []
    for keyword in keywords_list:
        keywords.append(keyword["name"])
    return keywords


def extract_data_from_file(data):
    movie_id = data["movielensId"]  # int
    tmdb_id = data["movielens"]["tmdbMovieId"]  # int
    imdb_link = get_str_item(data["imdb"]["imdbLink"])  # str
    languages = get_array_items(data["movielens"]["languages"])  # List[str]
    release_date = get_str_item(data["movielens"]["releaseDate"])
    release_date = datetime.strptime(release_date, DATETIME_FORMAT) if release_date is not None else None  # datetime
    release_year = get_release_year(data["movielens"]["releaseYear"])  # str
    runtime = get_num_item(data["movielens"]["runtime"])  # int
    youtube_trailer_video_ids = data["movielens"]["youtubeTrailerIds"]  # List[str]
    synopsis = get_str_item(data["movielens"]["plotSummary"])  # str
    num_ratings = get_num_item(data["movielens"]["numRatings"])  # int
    avg_ratings = get_num_item(data["movielens"]["avgRating"])  # float
    poster = get_poster(data["movielens"]["posterPath"], data["movielens"]["tmdbMovieId"])  # str
    title = get_str_item(data["movielens"]["originalTitle"])  # str
    genres = get_array_items(data["movielens"]["genres"])  # List[str]
    actors = get_array_items(data["movielens"]["actors"])  # List[str]
    directors = get_array_items(data["movielens"]["directors"])  # List[str]
    writers = get_array_items(data["imdb"]["writers"])  # List[str]
    age_rating = get_str_item(data["movielens"]["mpaa"])  # str

    try:
        tmdb_recommendations = data["tmdb"]["recommendations"]  # List[str], rec based on tmdb_id
        tmdb_keywords = get_keywords(data["tmdb"]["keywords"])  # List[str], keywords
        tmdb_popularity = get_num_item(data["tmdb"]["popularity"])  # float
    except KeyError:
        tmdb_recommendations = None
        tmdb_keywords = None
        tmdb_popularity = None
    except Exception as e:
        print("Unexpected exception encountered in extract_data_from_file:")
        print(e)
        tmdb_recommendations = None
        tmdb_keywords = None
        tmdb_popularity = None

    return {
        "movie_id": movie_id,
        "tmdb_id": tmdb_id,
        "title": title,
        "genres": genres,
        "age_rating": age_rating,
        "avg_ratings": avg_ratings,
        "num_ratings": num_ratings,
        "languages": languages,
        "actors": actors,
        "directors": directors,
        "writers": writers,
        "release_year": release_year,
        "release_date": release_date,
        "runtime": runtime,
        "imdb_link": imdb_link,
        "poster": poster,
        "youtube_trailer_video_ids": youtube_trailer_video_ids,
        "synopsis": synopsis,
        "tmdb_recommendations": tmdb_recommendations,
        "tmdb_keywords": tmdb_keywords,
        "tmdb_popularity": tmdb_popularity
    }


def extract_data(root: str) -> Tuple[int, int, int]:

    created = 0
    updated = 0
    files = os.listdir(root)
    for file_name in files:
        with open(os.path.join(root, file_name), "r", encoding="utf-8") as file:

            try:
                data = extract_data_from_file(json.load(file))
            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError encountered in file {file_name}")
                print(e)
                return len(files), updated, created

            movie, create = Movie.objects.update_or_create(movie_id=data.pop("movie_id"), defaults=data)

            if create:
                created += 1
            else:
                updated += 1

    return len(files), updated, created

