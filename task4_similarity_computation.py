
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
df = pd.read_csv("data/tmdb_movies_cleaned.csv")

# Load TF-IDF matrix from Task 3
tfidf_matrix = load_npz("data/tfidf_matrix.npz")

print("SIMILARITY COMPUTATION")
print("=" * 50)

print("Number of movies:", tfidf_matrix.shape[0])
print("Number of TF-IDF features:", tfidf_matrix.shape[1])

# Compute cosine similarity between all movies
similarity_matrix = cosine_similarity(tfidf_matrix)

# Display matrix shape
print("\nSIMILARITY MATRIX SHAPE")
print("=" * 50)
print(similarity_matrix.shape)

# Display a small sample of the matrix
print("\nFIRST 5 MOVIES SIMILARITY MATRIX")
print("=" * 50)
print(np.round(similarity_matrix[:5, :5], 3))

# Verify similarity values
print("\nSIMILARITY VALUE CHECK")
print("=" * 50)
print("Minimum similarity:", similarity_matrix.min())
print("Maximum similarity:", similarity_matrix.max())

# Save similarity matrix
np.save("data/similarity_matrix.npy", similarity_matrix)

print("\nSimilarity matrix saved successfully!")
print("Output file: data/similarity_matrix.npy")

# Display most similar movies for the first movie
if len(df) > 1:
    movie_index = 0

    similar_indices = np.argsort(
        similarity_matrix[movie_index]
    )[::-1][1:6]

    print("\nTOP 5 SIMILAR MOVIES")
    print("=" * 50)
    print("Selected movie:", df.iloc[movie_index]["title"])

    for index in similar_indices:
        print(
            df.iloc[index]["title"],
            "- Similarity:",
            round(similarity_matrix[movie_index][index], 4)
        )

print("\nTask 4 completed successfully!")
