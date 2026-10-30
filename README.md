# Movie Recommendation System

A content-based movie recommendation system built using Python, Pandas, Scikit-learn and Streamlit.

## Project Overview

This project recommends movies based on the similarity of their descriptions. It uses TF-IDF vectorization to convert movie overviews into numerical features and cosine similarity to identify similar movies.

## Features

- Movie selection through a dropdown menu.
- Content-based movie recommendations.
- TF-IDF text vectorization.
- Cosine similarity computation.
- Top 5 similar movie recommendations.
- Similarity scores for recommended movies.
- Interactive Streamlit interface.

## Dataset

**Dataset:** TMDB 5000 Movie Dataset

Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

The cleaned movie dataset is stored in `data/tmdb_movies_cleaned.csv`.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Streamlit
- Git and GitHub

## Project Structure

```text
GenAI-Task20-Recommendation/
│
├── data/
│   └── tmdb_movies_cleaned.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd GenAI-Task20-Recommendation
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## How It Works

1. Movie descriptions are cleaned during preprocessing.
2. TF-IDF converts the descriptions into numerical vectors.
3. Cosine similarity measures the similarity between the selected movie and other movies.
4. Movies are sorted by similarity score.
5. The application displays the top five recommendations.

## Recommendation Approach

This project uses content-based filtering. It recommends movies based on their textual descriptions rather than user ratings or collaborative filtering.

## Author

Praveg Chikte

## Project

Assignment 20 — Building and Deploying a Recommendation System