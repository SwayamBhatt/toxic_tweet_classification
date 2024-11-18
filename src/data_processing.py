import re
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    return df

def clean_text(text):
    """Cleans raw text data."""
    return re.sub(r"http\S+|@\w+|#\w+|[^\w\s]", "", text).lower().strip()

def preprocess_data(df):
    """Cleans and splits the dataset."""
    df['cleaned_tweet'] = df['tweet'].apply(clean_text)
    df.drop(['tweet'], axis=1, inplace=True)
    X = df['cleaned_tweet']
    y = df['Toxicity']
    return train_test_split(X, y, test_size=0.2, random_state=42)