
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("data/tmdb_movies_cleaned.csv")

# Handle any missing cleaned text
df["clean_text"] = df["clean_text"].fillna("")

# Configure TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

# Convert text into TF-IDF vectors
tfidf_matrix = tfidf.fit_transform(df["clean_text"])

# Display matrix shape
print("TF-IDF VECTORIZATION RESULTS")
print("=" * 50)

print("Dataset shape:", df.shape)
print("TF-IDF matrix shape:", tfidf_matrix.shape)

print("\nNumber of movies:", tfidf_matrix.shape[0])
print("Number of features:", tfidf_matrix.shape[1])

# Display first 10 feature names
print("\nFIRST 10 TF-IDF FEATURES")
print("=" * 50)
print(tfidf.get_feature_names_out()[:10])

# Display a sample TF-IDF vector
print("\nSAMPLE TF-IDF VECTOR")
print("=" * 50)
print(tfidf_matrix[0].toarray())

# Save TF-IDF matrix
from scipy.sparse import save_npz

save_npz("data/tfidf_matrix.npz", tfidf_matrix)

# Save feature names
pd.DataFrame({
    "feature": tfidf.get_feature_names_out()
}).to_csv("data/tfidf_features.csv", index=False)

print("\nTF-IDF matrix saved successfully!")
print("Feature names saved successfully!")
print("\nTask 3 completed successfully!")
