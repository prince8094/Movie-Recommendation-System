import streamlit as st
import pandas as pd
import joblib

from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Load Data
# ----------------------------

movies = joblib.load("movies.pkl")
embeddings = joblib.load("movie_embeddings.pkl")

movie_titles = movies["Title"].tolist()

# ----------------------------
# Recommendation Function
# ----------------------------

def recommend(movie_name, n=5):

    match = process.extractOne(
        movie_name,
        movie_titles,
        scorer=fuzz.WRatio,
        score_cutoff=70
    )

    if match is None:
        return None, None

    movie = match[0]

    idx = movies[movies["Title"] == movie].index[0]

    scores = cosine_similarity(
       embeddings[idx].reshape(1, -1),
       embeddings
    ).flatten()

    scores = list(enumerate(scores))

    scores = sorted(
      scores,
      key=lambda x: x[1],
      reverse=True
    )

    recommendations = []

    for i, score in scores[1:]:

        if score < 0.50:
            continue

        m = movies.iloc[i]

        recommendations.append({

            "Movie": m["Title"],
            "Similarity": round(score * 100, 2),
            "Genre": m["Genre"],
            "Rating": m["Vote_Average"],
            "Popularity": round(m["Popularity"],2),
            "Release Year": m["Release_Year"],
            "Poster": m["Poster_Url"]

        })

        if len(recommendations) == n:
            break

    return movie, recommendations

# ----------------------------
# UI
# ----------------------------

st.title("🎬 Movie Recommendation System")

st.markdown(
    "Discover movies similar to your favorite films using **Sentence-BERT Semantic Embeddings**."
)

movie_input = st.selectbox(
    "Select a Movie",
    sorted(movie_titles),
    index=None,
    placeholder="Search a movie..."
)

num = st.slider(
    "Number of Recommendations",
    1,
    10,
    5
)

if st.button("Recommend"):

    if movie_input == "":
        st.warning("Please enter a movie name.")
        st.stop()

    movie, recs = recommend(movie_input, num)

    if movie is None:
        st.error("Movie not found.")
        st.stop()

    st.success(f"Showing recommendations for **{movie}**")

    for rec in recs:

        col1, col2 = st.columns([1,3])

        with col1:

            st.image(
                rec["Poster"],
                use_container_width=True
            )

        with col2:

            st.subheader(rec["Movie"])

            st.write(f"🎭 Genre : {rec['Genre']}")

            st.write(f"⭐ Rating : {rec['Rating']}")

            st.write(f"📈 Popularity : {rec['Popularity']}")

            st.write(f"📅 Release Year : {rec['Release Year']}")

            st.progress(min(rec["Similarity"]/100,1.0))

            st.write(f"**Match : {rec['Similarity']}%**")

        st.divider()