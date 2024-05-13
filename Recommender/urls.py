from django.urls import path
from . import views

app_name = "Recommender"
urlpatterns = [
    path("", views.index, name="index"),
    path("recommend/user/", views.recommend_user, name="recommend_user"),
    path("recommend/movie/<int:movie_id>/", views.recommend_item, name="recommend_item"),
    path("recommend/movie/", views.recommend_item, name="recommend_item"),
    path("compute_vecs/", views.compute_synopsis_vecs, name="compute_synopsis_vecs"),
    path("extract_data/drive/", views.extract_drive_data, name="extract_drive_data"),

    path("search", views.search, name="search"),
    path("search_db", views.search_db, name="search_db"),
    # change path to a better name
    # also change it in search.html
    path("movierecommendations", views.movie_recommendations, name="movie_recommendations")
]
