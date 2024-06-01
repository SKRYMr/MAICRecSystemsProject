from django.urls import path
from . import views

app_name = "Recommender"
urlpatterns = [
    path("", views.index, name="index"),
    path("recommend/user/", views.recommend_user, name="recommend_user"),
    path("recommend/movie/<int:movie_id>/", views.neighbours_recommend, name="neighbours_recommend"),
    path("recommend/movie/", views.semantic_recommend, name="semantic_recommend"),
    path("compute_vecs/", views.compute_synopsis_vecs, name="compute_synopsis_vecs"),
    path("extract_data/drive/", views.extract_drive_data, name="extract_drive_data"),
    path("extract_data/kaggle/", views.extract_kaggle_posters, name="extract_kaggle_posters"),
    path("extract_data/imdb/", views.scrape_imdb_posters, name="extract_imdb_posters"),
    path("extract_data/ratings/", views.import_ratings, name="import_ratings"),

    path("search", views.search, name="search"),
    path("search_db", views.search_db, name="search_db"),
    # change path to a better name
    # also change it in search.html
    path("movierecommendations", views.movie_recommendations, name="movie_recommendations")
]
