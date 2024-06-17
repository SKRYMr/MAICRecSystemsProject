# Generated by Django 5.0.4 on 2024-05-28 13:59

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Movie',
            fields=[
                ('movie_id', models.IntegerField(primary_key=True, serialize=False)),
                ('tmdb_id', models.IntegerField(blank=True, null=True)),
                ('title', models.CharField(blank=True, max_length=200, null=True)),
                ('genres', models.TextField(blank=True, null=True)),
                ('age_rating', models.CharField(blank=True, max_length=50, null=True)),
                ('avg_ratings', models.FloatField(blank=True, null=True)),
                ('num_ratings', models.IntegerField(blank=True, null=True)),
                ('languages', models.TextField(blank=True, null=True)),
                ('actors', models.TextField(blank=True, null=True)),
                ('directors', models.TextField(blank=True, null=True)),
                ('writers', models.TextField(blank=True, null=True)),
                ('release_year', models.IntegerField(blank=True, null=True)),
                ('release_date', models.DateField(blank=True, null=True)),
                ('runtime', models.IntegerField(blank=True, null=True)),
                ('imdb_link', models.CharField(blank=True, max_length=500, null=True)),
                ('poster', models.CharField(blank=True, max_length=100, null=True)),
                ('youtube_trailer_video_ids', models.TextField(blank=True, null=True)),
                ('synopsis', models.TextField(max_length=5000, null=True)),
                ('synopsis_vec', models.TextField(null=True)),
                ('tmdb_recommendations', models.TextField(blank=True, null=True)),
                ('tmdb_keywords', models.TextField(blank=True, null=True)),
                ('tmdb_popularity', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('user_id', models.IntegerField(primary_key=True, serialize=False)),
            ],
        ),
        migrations.CreateModel(
            name='Rating',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rating', models.IntegerField(choices=[(1, 'Very Bad'), (2, 'Bad'), (3, 'Neutral'), (4, 'Good'), (5, 'Very Good')])),
                ('movie', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Recommender.movie')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Recommender.user')),
            ],
        ),
    ]
