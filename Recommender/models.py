from django.db import models


# Create your models here.
class Movie(models.Model):
    movie_id = models.IntegerField(primary_key=True)
    tmdb_id = models.IntegerField(null=True, blank=True)
    title = models.CharField(max_length=200, null=True, blank=True)
    genres = models.TextField(null=True, blank=True)
    age_ratings = models.CharField(max_length=50, null=True, blank=True)
    avg_ratings = models.FloatField(null=True, blank=True)
    num_ratings = models.IntegerField(null=True, blank=True)
    languages = models.TextField(null=True, blank=True)
    actors = models.TextField(null=True, blank=True)
    directors = models.TextField(null=True, blank=True)
    writers = models.TextField(null=True, blank=True)
    release_year = models.IntegerField(null=True, blank=True)
    release_date = models.DateField(null=True, blank=True)
    runtime = models.IntegerField(null=True, blank=True)
    imdb_link = models.CharField(max_length=500, null=True, blank=True)
    poster = models.CharField(max_length=100, null=True, blank=True)
    youtube_trailer_video_ids = models.TextField(null=True, blank=True)
    synopsis = models.TextField(max_length=5000, null=True)
    synopsis_vec = models.TextField(null=True)
    tmdb_recommendations = models.TextField(null=True, blank=True)

    @staticmethod
    def get_base_url():
        """
        Returns the base url for the movie posters, just append whatever is in the "poster" field.
        """
        return "https://image.tmdb.org/t/p/original"


class User(models.Model):
    user_id = models.IntegerField(primary_key=True)


class Rating(models.Model):
    RATINGS = {
        1: "Very Bad",
        2: "Bad",
        3: "Neutral",
        4: "Good",
        5: "Very Good"
    }
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=RATINGS)
