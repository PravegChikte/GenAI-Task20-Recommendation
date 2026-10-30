
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# Load and vectorize dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/tmdb_movies_cleaned.csv")
    df["clean_text"] = df["clean_text"].fillna("")

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(df["clean_text"])

    return df, tfidf_matrix


df, tfidf_matrix = load_data()


# Recommendation function
def recommend(item_name, top_n=5):

    indices = df[
        df["title"].fillna("").str.lower() == item_name.lower()
    ].index.tolist()

    if not indices:
        return []

    movie_index = indices[0]

    # Calculate similarity against all movies
    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort similarity scores
    similar_indices = scores.argsort()[::-1]

    # Exclude selected movie
    similar_indices = [
        i for i in similar_indices
        if i != movie_index
    ][:top_n]

    results = []

    for i in similar_indices:
        results.append({
            "Movie": df.iloc[i]["title"],
            "Similarity Score": round(float(scores[i]), 4)
        })

    return results


# Streamlit interface
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourite movie "
    "using a content-based recommendation system."
)

st.divider()

movie_list = sorted(
    df["title"].dropna().astype(str).unique().tolist()
)

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

if st.button("Get Recommendations", type="primary"):

    results = recommend(selected_movie, top_n=5)

    st.subheader(f"Movies similar to {selected_movie}")

    if results:
        for i, movie in enumerate(results, start=1):
            st.write(f"**{i}. {movie['Movie']}**")
            st.caption(
                f"Similarity Score: {movie['Similarity Score']}"
            )
    else:
        st.warning("No recommendations found.")

st.divider()

st.caption(
    "Powered by Python, Pandas, Scikit-learn and Streamlit"
)
