
import pandas as pd
import numpy as np

# Load movie dataset
df = pd.read_csv("data/tmdb_movies_cleaned.csv")

# Load similarity matrix
similarity_matrix = np.load("data/similarity_matrix.npy")

# Recommendation function
def recommend(item_name, top_n=5):

    # Find movie index
    movie_indices = df[
        df["title"].str.lower() == item_name.lower()
    ].index.tolist()

    # Check whether movie exists
    if not movie_indices:
        print(f"Movie '{item_name}' not found!")
        return []

    # Get index of selected movie
    movie_index = movie_indices[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    # Sort movies by similarity score (highest first)
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove selected movie itself
    similarity_scores = [
        item for item in similarity_scores
        if item[0] != movie_index
    ]

    # Select top N recommendations
    top_movies = similarity_scores[:top_n]

    # Display recommendations
    recommendations = []

    for index, score in top_movies:
        movie_title = df.iloc[index]["title"]

        recommendations.append({
            "Movie": movie_title,
            "Similarity Score": round(float(score), 4)
        })

    return recommendations


# Test recommendation function with 3 different movies
test_movies = [
    "Avatar",
    "Titanic",
    "The Dark Knight"
]

for movie in test_movies:

    print("\n" + "=" * 55)
    print("RECOMMENDATIONS FOR:", movie)
    print("=" * 55)

    results = recommend(movie, top_n=5)

    if results:
        for number, result in enumerate(results, start=1):
            print(
                f"{number}. {result['Movie']} "
                f"| Similarity: {result['Similarity Score']}"
            )

print("\nTask 5 completed successfully!")
