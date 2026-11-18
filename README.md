# Movie Recommendation System (Sentence-BERT)

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=flat-square&logo=streamlit)
![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-Semantic_Search-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Cosine_Similarity-orange?style=flat-square&logo=scikitlearn)

**Live app:** https://movie-recommendation-system-prince.streamlit.app/

## Overview

This is a movie recommendation app that ranks results by semantic similarity rather than keyword overlap. Instead of matching on shared words in a plot summary, it embeds each movie's overview with Sentence-BERT and compares meaning, so it can surface movies with similar themes or stories even when they're described in completely different language.

It's built as a Streamlit app. Search for a movie and you get back a ranked list of similar titles, each with poster, rating, genre, popularity, release year, and a match percentage.

## Screenshots

**Home page**

(add screenshot here)

**Recommendation output**

(add screenshot here)

## Why not TF-IDF

Most simple recommenders lean on TF-IDF or plain keyword matching. That works fine when two descriptions share vocabulary, but it breaks down fast:

- Two movies with the same story but different wording won't match.
- A typo or partial title returns nothing.
- Matches often come from one shared word rather than actual plot similarity.

Search for "Avatar" and you'd expect results like *Avatar 2*, *Pacific Rim*, *Moonfall*, *Inuyashiki*, or *Cosmic Sin* — not something that only happens to share the word "science fiction" in its description. Getting that right meant moving past word overlap and toward something that understands what a movie is actually about.

## How it works

1. The user types a movie title, which is corrected against the dataset using RapidFuzz fuzzy matching (so `avatarr`, `AVATAR`, or a partial title all resolve correctly).
2. The app loads the precomputed embedding for the matched movie.
3. Cosine similarity is computed between that embedding and every other movie's embedding, on the fly.
4. Anything below the similarity threshold (0.50 by default) is dropped.
5. The remaining results are sorted and the top N are shown.

The embedding model is `all-MiniLM-L6-v2`, which turns each movie overview into a 384-dimensional vector. Because it's trained for semantic similarity rather than word frequency, it captures storyline and concept similarity that TF-IDF misses entirely.

## Tech stack

| Category | Tool |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Embeddings | Sentence-BERT |
| Similarity | Cosine similarity (scikit-learn) |
| Fuzzy search | RapidFuzz |
| Data handling | Pandas |
| Serialization | Joblib |

## Dataset

The dataset includes title, overview, genres, release year, vote average, popularity, and poster URL for each movie. Every overview is embedded ahead of time and stored so the app doesn't need to re-run the model at request time.

## Project structure

```
Movie-Recommendation-System/
├── app.py
├── movies.pkl
├── movie_embeddings.pkl
├── mymoviedb.csv
├── Movie_Recommendation_BERT.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
git clone https://github.com/prince8094/Movie-Recommendation-System.git
cd Movie-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```

## Notable problems along the way

**TF-IDF recommendations were weak.** The first version used TF-IDF, and results were often only loosely related — movies would match on a shared word with nothing else in common. Switching to Sentence-BERT embeddings fixed this by comparing meaning instead of vocabulary.

**The similarity matrix was too big to ship.** An early version precomputed and stored a full pairwise similarity matrix (`similarity.pkl`), which came out to roughly 368 MB — too large for a clean GitHub repo or a fast Streamlit deploy. Removing it and computing cosine similarity on the fly from the stored embeddings cut the repo size dramatically and made deployment far simpler.

**Search required an exact title.** Typing anything other than the precise movie name returned no results. RapidFuzz fixed this by fuzzy-matching user input against the dataset, so close or partial titles resolve correctly.

**Reruns were slow.** Streamlit was reloading the model and embeddings on every interaction. Wrapping the load calls in `@st.cache_resource` keeps them in memory across reruns instead of hitting disk each time.

## Notes on evaluation

This is a content-based recommender, not a classifier, so there's no accuracy metric in the usual sense. Quality is judged by inspection — whether results share theme, story, or genre with the query — and the move from TF-IDF to Sentence-BERT was a clear, visible improvement on that front, especially for movies that tell similar stories in very different language.

## Possible next steps

- Hybrid recommendations combining content-based and collaborative filtering
- User accounts with watch history and saved favorites
- Trailer integration via the TMDB API
- Genre and actor-based filtering

## Author

**Prince Gupta** — B.Tech CSE, focused on data science and machine learning.

GitHub: https://github.com/prince8094
LinkedIn: https://www.linkedin.com/in/prince-gupta-8a285a328/

If this project is useful to you, a star on the repo is appreciated.
