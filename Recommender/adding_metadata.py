import pandas as pd
import numpy as np
import json
import ast


def extract_3_main_actors(string):
    try:
        cast = ast.literal_eval(string)
    except ValueError:
        return []


    if len(cast) >= 3:
        return [(item['character'], item['name']) for item in cast[:3]]
    else:
        return [(item['character'], item['name']) for item in cast]


def extract_director(string):
    try:
        crew = ast.literal_eval(string)
    except ValueError:
        return ''

    for i in crew:
        if i['job'] == 'Director':
            return i['name']
    return ''



# This script requires movies_metadata.csv, credits.csv and links.csv file.
# These files were processed in this script and matched with existing columns
# They were downloaded from https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset and are not included in the repository

# Step 1: filtering the links to include only the ones we include in the movies dataset

# movies = pd.read_csv("./data/movies.csv")
# links = pd.read_csv("./data/links.csv")
# print(links.shape)
#
# links = links.rename(columns={"movieId": "movie_id"})
#
# merged = pd.merge(movies, links, on='movie_id', how='left')
#
# print(merged.shape)
# print(merged.head())
#
# merged.tmdbId = merged.tmdbId.fillna(-1)
#
# links_filtered = merged[["movie_id", "tmdbId"]]
#
# links_filtered["tmdbId"] = links_filtered.tmdbId.astype(np.int64)
#
# links_filtered.to_csv("./data/links_filtered.csv", index=False)

# Step 2: Filter the credits to include only the movies we need

# links = pd.read_csv("./data/links_filtered.csv")
# credits = pd.read_csv("./data/credits.csv")
# print(credits.shape)
#
# credits = credits.rename(columns={"id": "tmdbId"})
#
# merged = pd.merge(links, credits, on='tmdbId', how='left')
#
# print(merged.shape)
# print(merged.head())
# print(merged.cast.head())
#
# merged.to_csv("./data/credits_filtered.csv", index=False)


# Step 3: Extract cast and director from credits csv

# credits = pd.read_csv("./data/credits_filtered.csv")
#
# credits.cast = credits.cast.apply(extract_3_main_actors)
#
# credits.crew = credits.crew.apply(extract_director)
#
# credits = credits.rename(columns={"crew": "director"})
#
# credits.to_csv("./data/credits_filtered.csv", index=False)

# Step 4 Filter metadata to include only movies/fields

# metadata = pd.read_csv("./data/movies_metadata.csv",usecols=["id", "overview","poster_path"])
# links = pd.read_csv("./data/links_filtered.csv")
#
# print(metadata.shape)
#
# metadata = metadata.rename(columns={"id": "tmdbId", "overview":"synopsis","poster_path":"poster"})
# metadata = metadata[["poster", "tmdbId", "synopsis"]]
#
# metadata.tmdbId = metadata.tmdbId[metadata.tmdbId.str.isnumeric()]
#
# metadata.tmdbId = metadata.tmdbId.fillna(-1)
#
# metadata["tmdbId"] = metadata.tmdbId.astype(np.int64)
#
# merged = pd.merge(links, metadata, on='tmdbId', how='left')
#
# print(merged.shape)
# print(merged.head())
#
# merged.to_csv("./data/metadata_filtered.csv", index=False)

# Step 5 Merging movies with metadata

# metadata = pd.read_csv("./data/metadata_filtered.csv")
# credits = pd.read_csv("./data/credits_filtered.csv")
# movies = pd.read_csv("./data/movies.csv")
#
# merged = pd.merge(movies, credits, on='movie_id', how='left')
# merged = merged.drop(columns=["tmdbId"])
# movies1 = pd.merge(merged, metadata, on='movie_id', how='left')
# movies1 = movies1.drop(columns=["tmdbId","Unnamed: 0"])
# movies1 = movies1.rename(columns={"cast": "actors"})
#
# movies1['year'] = movies1['title'].str[-5:-1]
# movies1.title = movies1.title.str[:-7]
#
# movies1 = movies1[["movie_id", "title","director", "actors", "synopsis", "genres", "poster","year"]]
#
# movies1.to_csv("./data/movies_ext.csv", index=False)

# Step 6 Aligning the posters of movies_ext.csv

posters = pd.read_csv("./data/metadata_filtered.csv",usecols=["movie_id", "poster"])
movies = pd.read_csv("./data/movies_ext.csv")

merged = pd.merge(movies, posters, on='movie_id', how='left')
merged["poster"] = merged["poster_y"]
merged = merged[["movie_id", "title","director", "actors", "synopsis", "genres", "poster","year"]]

merged = merged.drop_duplicates(subset=['movie_id'])

merged.to_csv("./data/movies_ext.csv",index=False)














