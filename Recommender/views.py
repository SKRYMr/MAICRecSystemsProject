import pandas as pd
from django_pandas.io import read_frame
from django.contrib.auth.decorators import user_passes_test
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render
from tqdm import tqdm
from .extract_data import extract_data, GOOGLE_DRIVE_ROOT, extract_posters, POSTERS_CSV_PATH, RATINGS_CSV_PATH
from .models import User, Movie, Rating

from .utils import (scrape_imdb_poster, get_movies_recommendations, compute_synopsis_vec,
                    format_movie_recommendations, evaluate_recommendations)
from .movie_rec_methods import (tqdm_recommendations, gpt_recommendations, year_genre_recommend,
                                neighbours_recommend, semantic_recommend)


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
def compute_neighbours_recommend(request):
    force = request.GET.get("force", False)


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


@user_passes_test(lambda u: u.is_superuser)
def scrape_imdb_posters(request, batch_size: int = 100, safe: bool = True, force: bool = False):
    movies = Movie.objects.all()
    total = len(movies)
    success = 0
    errors = 0
    ignored = 0
    objs = []
    for i, movie in tqdm(enumerate(movies)):
        if not force and movie.imdb_poster:
            ignored += 1
            continue
        if safe and errors >= 100:
            break
        try:
            if not movie.imdb_link:
                print(f"Movie {movie.movie_id} has no IMDB link.")
                errors += 1
                continue
            movie.imdb_poster = scrape_imdb_poster(movie.imdb_link)
            objs.append(movie)
        except Exception as e:
            print(f"Movie {movie.movie_id} encountered exception: {e}")
            errors += 1
            continue
        if len(objs) >= batch_size:
            with transaction.atomic():
                updated = Movie.objects.bulk_update(objs, ["imdb_poster"], batch_size)
                if updated > 0:
                    objs = []
                    success += updated
    with transaction.atomic():
        updated = Movie.objects.bulk_update(objs, ["imdb_poster"], batch_size)
        success += updated
    return render(request, "success.html",
                  {"context": {"total": total, "successful": success, "errors": errors, "ignored": ignored}})


@user_passes_test(lambda u: u.is_superuser)
def import_ratings(request, batch_size: int = 1000):
    df = pd.read_csv(RATINGS_CSV_PATH,
                     usecols=["userId", "movieId", "rating"]).rename(columns={"userId": "user_id",
                                                                              "movieId": "movie_id"})
    with transaction.atomic():
        deleted, _ = Rating.objects.all().delete()
    new_users = []
    new_ratings = []
    created_users = 0
    created_ratings = 0
    total_ratings = df.shape[0]
    for _, item in tqdm(df.iterrows(), total=total_ratings):
        if Movie.objects.filter(movie_id=item["movie_id"]).exists():
            if not User.objects.filter(user_id=item["user_id"]).exists():
                new_users.append(User(user_id=item["user_id"]))
            new_ratings.append(Rating(user_id=item["user_id"], movie_id=item["movie_id"], rating=item["rating"]))
        if len(new_ratings) >= batch_size:
            with transaction.atomic():
                newly_created_users = len(User.objects.bulk_create(new_users,
                                                                   batch_size=batch_size,
                                                                   ignore_conflicts=True)) if len(new_users) > 0 else 0
                newly_created_ratings = len(Rating.objects.bulk_create(new_ratings,
                                                                       batch_size=batch_size,
                                                                       ignore_conflicts=True))
                new_users = [] if newly_created_users > 0 else new_users
                new_ratings = [] if newly_created_ratings > 0 else new_ratings
                created_users += newly_created_users
                created_ratings += newly_created_ratings
    with transaction.atomic():
        created_users += len(User.objects.bulk_create(new_users, batch_size=batch_size, ignore_conflicts=True))
        created_ratings += len(Rating.objects.bulk_create(new_ratings, batch_size=batch_size, ignore_conflicts=True))
    return render(request, "success.html", {"context": {"deleted": deleted,
                                                        "total ratings": total_ratings,
                                                        "users created": created_users,
                                                        "ratings created": created_ratings}})


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
            "year": target_movie.release_year,
            "imdb_poster": target_movie.imdb_poster,
            "avg_ratings": target_movie.avg_ratings,
            "age_rating": target_movie.age_rating,
            "genres": target_movie.genres.replace("[", "").replace("]", "").replace("'", "").split(",") if target_movie.genres else [],
            "actors": ", ".join(target_movie.actors.replace("[", "").replace("]", "").replace("'", "").split(",")[:4]) if target_movie.actors else "",
            "directors": ", ".join(target_movie.directors.replace("[", "").replace("]", "").replace("'", "").split(",")[:4]) if target_movie.directors else "",
            # We can add more data if we want to I guess?
        }

        # TODO: Maybe we can use multiprocess to perform every call in parallel and save a little bit of time

        context = {
            "target_movie": target_movie_dict,
            "recommendations": {
                "TQDM Recommendations": tqdm_recommendations(movie_id),
                "ChatGPT Recommendations": gpt_recommendations(movie_id),
                "Year-Genre-Keywords Recommendations": year_genre_recommend(movie_id, metric="keyword"),
                "Year-Genre-Actor Recommendations": year_genre_recommend(movie_id, metric="actors"),
                "Neighbourhood Recommendations": neighbours_recommend(movie_id),
                "Semantic Similarity Recommendations": semantic_recommend(movie_id, scale=10)
            },
            "metrics": {
                "Year-Genre-Keywords Recommendations": "Score",
                "Year-Genre-Actor Recommendations": "Score",
                "Neighbourhood Recommendations": "Average Rating",
                "Semantic Similarity Recommendations": "Cosine Similarity"
            }
        }

        print(evaluate_recommendations(context['recommendations'], target_movie.tmdb_recommendations))

        return render(request, "movie_recommendations.html", context)
    else:
        context = {"error": f"Not a post request"}
        return render(request, "error.html", context)
