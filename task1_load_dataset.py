
import pandas as pd

# Load dataset
file_path = "data/tmdb_5000_movies.csv"
df = pd.read_csv(file_path)

# 1. Dataset shape
print("DATASET SHAPE")
print("=" * 50)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\nCOLUMN NAMES")
print("=" * 50)
print(df.columns.tolist())

# 3. First five rows
print("\nFIRST FIVE ROWS")
print("=" * 50)
print(df.head())

# 4. Dataset information
print("\nDATASET INFORMATION")
print("=" * 50)
df.info()

# 5. Missing values
print("\nMISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# 6. Identify text columns
print("\nTEXT COLUMNS FOR RECOMMENDATIONS")
print("=" * 50)

text_columns = ["title", "overview", "genres", "keywords"]

for column in text_columns:
    if column in df.columns:
        print(f"{column}: {df[column].dtype}")

print("\nTask 1 completed successfully.")
