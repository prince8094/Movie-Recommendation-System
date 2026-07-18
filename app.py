import streamlit as st
import pandas as pd
import joblib

from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_resource
def load_data():
    movies = joblib.load("movies.pkl")
    embeddings = joblib.load("movie_embeddings.pkl")
    return movies, embeddings

movies, embeddings = load_data()

movie_titles = movies["Title"].tolist()

with st.sidebar:

    st.title("🎬 Movie Recommender")

    st.write("Built using")

    st.markdown("""
- Sentence-BERT
- Cosine Similarity
- RapidFuzz
- Streamlit
""")
    
    st.markdown("---")

    st.caption(
       "Developed by Prince Gupta | Sentence-BERT Movie Recommendation System"
    )

# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------

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

    scores.sort(key=lambda x: x[1], reverse=True)

    recommendations = []

    for i, score in scores[1:]:

        if score < 0.50:
            continue

        m = movies.iloc[i]

        recommendations.append({

            "Movie": str(m["Title"]),
            "Similarity": float(round(float(score) * 100, 2)),
            "Genre": str(m["Genre"]),
            "Rating": float(m["Vote_Average"]),
            "Popularity": float(round(float(m["Popularity"]), 2)),
            "Release Year": int(m["Release_Year"]),
            "Poster": str(m["Poster_Url"])

        })

        if len(recommendations) == n:
            break

    return movie, recommendations

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🎬 Movie Recommendation System")

st.markdown(
    """
Discover movies similar to your favorites using
**Sentence-BERT Semantic Embeddings**.
"""
)

movie_input = st.selectbox(
    "🎥 Search Movie",
    sorted(movie_titles),
    index=None,
    placeholder="Type or search a movie..."
)

num = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# --------------------------------------------------
# Recommendation Button
# --------------------------------------------------

if st.button("🚀 Recommend"):

    if movie_input is None:
        st.warning("Please select a movie.")
        st.stop()

    movie, recs = recommend(movie_input, num)

    if movie is None:
        st.error("Movie not found.")
        st.stop()

    st.success(f"Showing recommendations for **{movie}**")

    if len(recs) == 0:
        st.warning("No highly similar movies found.")
        st.stop()

    for rec in recs:

        col1, col2 = st.columns([1, 3])

        with col1:

            st.image(
                rec["Poster"],
                use_container_width=True
            )

        with col2:

            st.subheader(rec["Movie"])

            st.write(f"🎭 **Genre:** {rec['Genre']}")

            st.write(f"⭐ **Rating:** {rec['Rating']:.1f}/10")

            st.write(f"📈 **Popularity:** {rec['Popularity']:.2f}")

            st.write(f"📅 **Release Year:** {rec['Release Year']}")

            progress = float(min(rec["Similarity"] / 100.0, 1.0))

            st.progress(progress)

            st.metric(
                label="🎯 Match Score",
                value=f"{rec['Similarity']:.2f}%"
            )

        st.divider()