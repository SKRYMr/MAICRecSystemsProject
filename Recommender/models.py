from django.db import models


class Movie(models.Model):
    movie_id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=500)
    director = models.CharField(max_length=100, null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    actors = models.TextField(null=True, blank=True)        # Actually a JSON encoded list of actors
    synopsis = models.TextField(max_length=5000, null=True, blank=True)
    synopsis_vec = models.TextField(null=True, blank=True)  # Actually a JSON encoded list of vector components
    genres = models.TextField(null=True, blank=True)        # Actually a JSON encoded list of genres
    poster = models.CharField(max_length=100, null=True, blank=True)

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
