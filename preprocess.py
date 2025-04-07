import pandas as pd

def load_dataset():
    try:
        df = pd.read_excel("D:\Clg\B.Tech\news-recommender\news-recommender\news_data_repository.xlsx")
        df = df.dropna(subset=["headline", "content"])  # Drop rows with missing essential data
        df["text"] = df["headline"] + " " + df["content"]
        df["category"] = df["type"]  # Create a 'category' column for filtering
        df["short_description"] = df["description"].fillna("No description available")
        return df
    except Exception as e:
        print("Error loading dataset:", e)
        return None
