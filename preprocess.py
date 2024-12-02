import pandas as pd
import re
import string
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Attempt to download resources
# this was from debugging
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download NLTK data (only needed once)
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_imdb_data(dataset_path):
    """
    Function to load, preprocess, and clean the IMDB dataset.
    Args:
        dataset_path (str): Path to the dataset file (CSV format).
    Returns:
        tuple: Cleaned and preprocessed training and testing datasets (X_train, X_test, y_train, y_test).
    """

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Check data structure
    print("Dataset loaded successfully. Sample rows:")
    print(df.head())

    # Standardize column names if necessary
    if 'sentiment' not in df.columns:
        df.columns = ['review', 'sentiment']

    # Remove duplicates and nulls
    print("Removing duplicates and null values...")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Convert sentiment labels to binary (0 = negative, 1 = positive)
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

    # Display examples of unprocessed reviews
    # Delete this
    print("\nExamples of unprocessed reviews:")
    print("Positive review:")
    print(df[df['sentiment'] == 1]['review'].iloc[0])
    print("Negative review:")
    print(df[df['sentiment'] == 0]['review'].iloc[0])

    # Clean text data
    print("\nCleaning text data...")
    def clean_text(text):
        text = text.lower()  # Lowercase text
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\[.*?\]', '', text)  # Remove text within brackets
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
        text = text.strip()  # Remove leading/trailing whitespace
        return text

    df['review'] = df['review'].apply(clean_text)

    # Remove stopwords and tokenize
    print("Removing stopwords and tokenizing...")
    stop_words = set(stopwords.words('english'))
    def tokenize_and_remove_stopwords(text):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)

    df['review'] = df['review'].apply(tokenize_and_remove_stopwords)

    # Display examples of processed reviews
    # Delete this
    print("\nExamples of processed reviews:")
    print("Positive review:")
    print(df[df['sentiment'] == 1]['review'].iloc[0])
    print("Negative review:")
    print(df[df['sentiment'] == 0]['review'].iloc[0])

    # Split data into training and testing sets
    print("\nSplitting dataset into train and test sets...")
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Preprocessing complete.")
    return X_train, X_test, y_train, y_test

# Usage example
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

if path is None:
    raise ValueError("Failed to download dataset. Ensure `kagglehub` is correctly configured.")
dataset_path = f"{path}/IMDB Dataset.csv"  # Adjust path to your dataset

X_train, X_test, y_train, y_test = preprocess_imdb_data(dataset_path)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
