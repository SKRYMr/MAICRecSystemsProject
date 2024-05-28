import pandas as pd
from django_pandas.io import read_frame
from django.contrib.auth.decorators import user_passes_test
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render
from .extract_data import extract_data, GOOGLE_DRIVE_ROOT, extract_posters, POSTERS_CSV_PATH
from .models import User, Movie, Rating

from .utils import get_movies_recommendations, compute_synopsis_vec, format_movie_recommendations, compare_age_rating
from .movie_rec_methods import tqdm_recommendations, gpt_recommendations, year_genre_recommend, neighbours_recommend, semantic_recommend

from typing import Literal, List


def index(request):
    return render(request, "search.html")


@user_passes_test(lambda u: u.is_superuser)
@transaction.atomic
def compute_synopsis_vecs(request):
    force = request.GET.get("force", True)
    movies = Movie.objects.all() if force else Movie.objects.filter(synopsis_vec__isnull=True)
    count = 0
    total = len(movies)
    for movie in movies:
        synopsis = movie.synopsis
        if synopsis is None:
            continue
        movie.synopsis_vec = compute_synopsis_vec(synopsis)
        if movie.synopsis_vec is not None:
            movie.synopsis_vec = movie.synopsis_vec.dumps().hex()
        movie.save()
        count += 1
    return render(request, "success.html", {"context": {"total": total, "count": count}})


@user_passes_test(lambda u: u.is_superuser)
@transaction.atomic
def extract_drive_data(request):
    total, created, updated = extract_data(GOOGLE_DRIVE_ROOT)
    return render(request, "success.html", {"context": {"total": total, "created": created, "updated": updated}})


@user_passes_test(lambda u: u.is_superuser)
@transaction.atomic
def extract_kaggle_posters(request):
    total, updated = extract_posters(POSTERS_CSV_PATH)
    return render(request, "success.html", {"context": {"total": total, "updated": updated}})


def recommend_user(request):
    users = set(User.objects.values_list("user_id", flat=True))
    movies = read_frame(Movie.objects.all())
    ratings = pd.DataFrame(list(Rating.objects.values_list("movie_id", "user_id", "rating")),
                           columns=["movie_id", "user_id", "rating"])

    try:
        # Check if user_id is in users
        if int(request.POST["user_id"]) not in users:
            context = {"error": f"User: {request.POST['user_id']} not found"}
            return render(request, "error.html", context)
    except ValueError:
        # user_id is not a number
        context = {"error": f"User id must be a number"}
        return render(request, "error.html", context)
    except Exception as e:
        print(e)
        context = {"error": f"An unexpected error has occurred"}
        return render(request, "error.html", context)

    try:
        user_predictions = get_movies_recommendations(request.POST["user_id"], users, ratings, movies,
                                                      minimum_ratings=3,
                                                      metric="neighbours_average")

        user_predictions = format_movie_recommendations(user_predictions, 20)

        context = {"recommendations": user_predictions.to_dict('records'), "user_id": request.POST['user_id']}
        return render(request, "recommendations.html", context)

    except Exception as e:
        print(e)
        context = {"error": f"An unexpected error has occurred"}
        return render(request, "error.html", context)


def search(request):
    return render(request, "search.html")


def search_db(request):
    # Columns to keep
    items = ["movie_id", "title", "poster"]

    search_term = request.GET["query"]
    offset = int(request.GET["offset"])

    if search_term:
        movies = Movie.objects.filter(title__icontains=search_term)
    else:
        movies = Movie.objects.all()

    movies = movies[offset:offset + 50]
    movies = read_frame(movies)
    movies = movies[items]

    return JsonResponse({"movies": movies.to_dict('records')})


# For 5 different methods of recommendations
# Add methods in movie_rec_methods.py
def movie_recommendations(request):
    if request.method == "POST":
        movie_id = request.POST["movie_id"]

        # Convert this to dict and add to context
        target_movie = Movie.objects.get(movie_id=movie_id)
        target_movie_dict = {
            "title": target_movie.title,
            "year": target_movie.release_year
            # We can add more data if we want to I guess? but title should be sufficient.
        }

        context = {
            "target_movie": target_movie_dict,
            "recommendations": {
                "TQDM Recommendations": tqdm_recommendations(movie_id),
                "ChatGPT Recommendations": gpt_recommendations(movie_id),
                "Year-Genre-Keywords Recommendations": year_genre_recommend(movie_id, type="keyword"),
                "Year-Genre-Actor Recommendations": year_genre_recommend(movie_id, type="actors"),
                "Neighbourhood Recommendations": neighbours_recommend(movie_id),
                "Semantic Similarity Recommendations": semantic_recommend(movie_id)
            },
            "metrics": {
                "Year-Genre-Keywords Recommendations": "Score",
                "Year-Genre-Actor Recommendations": "Score",
                "Neighbourhood Recommendations": "Average Rating",
                "Semantic Similarity Recommendations": "Cosine Similarity"
            }
        }

        return render(request, "movie_recommendations.html", context)
    else:
        context = {"error": f"Not a post request"}
        return render(request, "error.html", context)
