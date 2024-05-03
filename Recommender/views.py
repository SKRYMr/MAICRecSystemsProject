import pandas as pd
from django.http import Http404
from django_pandas.io import read_frame
from django.shortcuts import render
from .main import get_movies_recommendations
from .models import User, Movie, Rating
# Create your views here.


def index(request):
    return render(request, "index.html")


def recommend(request):
    users = set(User.objects.values_list("user_id", flat=True))
    movies = read_frame(Movie.objects.all())
    ratings = pd.DataFrame(list(Rating.objects.values_list("movie_id", "user_id", "rating")),
                           columns=["movie_id", "user_id", "rating"])

    try:
        # Check if user_id is in users
        if int(request.POST["user_id"]) not in users:
            context = {"error": f"User: {request.POST["user_id"]} not found"}
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

        user_predictions["actors"] = user_predictions["actors"].apply(lambda x: x.replace("[", "").replace("(", "").replace("'", "").replace("]", "").replace(")", "").split(", "))
        user_predictions["genres"] = user_predictions["genres"].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", "").split(", "))
        user_predictions["poster"] = user_predictions["poster"].apply(lambda x: "https://image.tmdb.org/t/p/original" + x)
        user_predictions["rating"] = user_predictions["rating"].apply(lambda x: round(x, 1))

        context = {"recommendations": user_predictions.to_dict('records')}
        return render(request, "recommendations.html", context)

    except Exception as e:
        print(e)
        context = {"error": f"An unexpected error has occurred"}
        return render(request, "error.html", context)

