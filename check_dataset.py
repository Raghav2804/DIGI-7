import pandas as pd

FILE_NAME = "tweets.csv"

try:
    df = pd.read_csv(FILE_NAME)
    print("\nDataset loaded successfully.")
    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
except FileNotFoundError:
    print(f"Error: {FILE_NAME} was not found.")
    print("Keep check_dataset.py and tweets.csv in the same folder.")
except Exception as e:
    print("Could not read the dataset:", e)
