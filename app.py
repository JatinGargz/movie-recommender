import streamlit as st
import pandas as pd
import numpy as np
import ast
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Smart Movie Recommender")

movies = pd.read_csv("https://raw.githubusercontent.com/tmdb-5000-movie-dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("https://raw.githubusercontent.com/tmdb-5000-movie-dataset/tmdb_5000_credits.csv")

tmdb = movies.merge(credits, on="title")

tmdb = tmdb[['title', 'overview', 'genres',
             'cast', 'crew', 'production_companies',
             'popularity', 'vote_average', 'vote_count', 'revenue']]

tmdb.dropna(inplace=True)
tmdb.reset_index(drop=True, inplace=True)

def convert(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'].replace(" ", "").lower())
    except:
        pass
    return L

def get_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return i['name'].replace(" ", "").lower()
    except:
        return ""
    return ""

tmdb['genres'] = tmdb['genres'].apply(convert)
tmdb['cast'] = tmdb['cast'].apply(lambda x: convert(x)[:3])
tmdb['production_companies'] = tmdb['production_companies'].apply(convert)
tmdb['director'] = tmdb['crew'].apply(get_director)

tmdb['overview'] = tmdb['overview'].apply(lambda x: x.lower())

def create_tags(row):
    return " ".join(row['genres']) + " " + \
           " ".join(row['cast']) + " " + \
           row['director'] + " " + \
           " ".join(row['production_companies']) + " " + \
           row['overview']

tmdb['tags'] = tmdb.apply(create_tags, axis=1)

tfidf = TfidfVectorizer(max_features=6000, stop_words='english')
vectors = tfidf.fit_transform(tmdb['tags'])

similarity = cosine_similarity(vectors)

tmdb['popularity'] = (tmdb['popularity'] - tmdb['popularity'].min()) / \
                     (tmdb['popularity'].max() - tmdb['popularity'].min())

tmdb['revenue'] = tmdb['revenue'].fillna(0)
tmdb['revenue'] = (tmdb['revenue'] - tmdb['revenue'].min()) / \
                  (tmdb['revenue'].max() - tmdb['revenue'].min())

C = tmdb['vote_average'].mean()
m = tmdb['vote_count'].quantile(0.70)

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(v+m) * C)

tmdb['weighted_rating'] = tmdb.apply(weighted_rating, axis=1)

tmdb['weighted_rating'] = (tmdb['weighted_rating'] - tmdb['weighted_rating'].min()) / \
                          (tmdb['weighted_rating'].max() - tmdb['weighted_rating'].min())

def recommend(movie_name, top_n=5):
    titles = tmdb['title'].tolist()
    match = get_close_matches(movie_name, titles, n=1)

    if not match:
        return []

    movie_name = match[0]
    idx = tmdb[tmdb['title'] == movie_name].index[0]

    target = tmdb.iloc[idx]

    target_genres = set(target['genres'])
    target_cast = set(target['cast'])
    target_director = target['director']

    scores = []

    for i, row in tmdb.iterrows():
        if i == idx:
            continue

        row_genres = set(row['genres'])

        if not (row_genres & target_genres):
            continue

        score = 0
        reason = []

        genre_overlap = len(row_genres & target_genres)
        score += genre_overlap * 5
        reason.append("Strong Genre Match")

        if set(row['cast']) & target_cast:
            score += 2
            reason.append("Same Actor")

        if row['director'] == target_director:
            score += 2
            reason.append("Same Director")

        sim = similarity[idx][i]

        final_score = (
            (score * 0.6) +
            (sim * 0.25) +
            (row['weighted_rating'] * 0.1) +
            (row['popularity'] * 0.05)
        )

        scores.append((row['title'], final_score, reason, row['genres']))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores[:top_n]

movie = st.text_input("Enter Movie Name")

if st.button("Recommend"):
    results = recommend(movie)

    if results:
        for i, (title, score, reason, genres) in enumerate(results, 1):
            st.write(f"{i}. {title}")
            st.write(f"Genres: {genres}")
            st.write(f"Reason: {reason}")
            st.write("---")
    else:
        st.write("Movie not found")
