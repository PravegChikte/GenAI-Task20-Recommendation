
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download English stopwords
nltk.download("stopwords", quiet=True)

# Load dataset
df = pd.read_csv("data/tmdb_5000_movies.csv")

print("Original dataset shape:", df.shape)

# Handle missing values
df["overview"] = df["overview"].fillna("")

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Text preprocessing function
def clean_text(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Join words back into a sentence
    text = " ".join(words)

    return text


# Apply preprocessing to overview column
df["clean_text"] = df["overview"].apply(clean_text)

# Display original and cleaned text
print("\nORIGINAL VS CLEANED TEXT")
print("=" * 60)

for i in range(min(5, len(df))):
    print(f"\nMovie {i + 1}")
    print("Original:", df["overview"].iloc[i])
    print("Cleaned :", df["clean_text"].iloc[i])
    print("-" * 60)

# Check missing values
print("\nMISSING VALUE CHECK")
print("=" * 60)
print("Original overview missing values:", df["overview"].isnull().sum())
print("Clean text missing values:", df["clean_text"].isnull().sum())

# Save cleaned dataset
df.to_csv("data/tmdb_movies_cleaned.csv", index=False)

print("\nDATASET INFORMATION")
print("=" * 60)
print("Dataset shape:", df.shape)
print("New column added: clean_text")
print("Saved file: data/tmdb_movies_cleaned.csv")

print("\nTask 2 completed successfully!")
