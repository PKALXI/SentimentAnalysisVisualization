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

def clean_text(text):
    """
    Cleans raw text by lowercasing, removing HTML tags, URLs, punctuation, and extra spaces.
    """
    text = text.lower()  # Lowercase text
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within brackets
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = text.strip()  # Remove leading/trailing whitespace
    return text

def tokenize_and_remove_stopwords(text, stop_words):
    """
    Tokenizes text and removes stopwords.
    """
    tokens = word_tokenize(text) # Tokenize text into words
    filtered_tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
    return ' '.join(filtered_tokens) # Rejoin tokens into a single string

def preprocess_imdb_data(dataset_path):
    """
    Function to load, preprocess, and clean the IMDB dataset.
    Args:
        dataset_path (str): Path to the dataset file (CSV format).
    Returns:
        tuple: Cleaned and preprocessed training and testing datasets (X_train, X_test, y_train, y_test).
    """
    print("Loading dataset...")
    df = pd.read_csv(dataset_path) # Load dataset from CSV
    
    print("Dataset loaded successfully. Sample rows:")
    print(df.head()) # Display first few rows

    # Standardize column names
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
    print("\nExamples of unprocessed reviews:")
    print("Positive review:", df[df['sentiment'] == 1]['review'].iloc[0])
    print("Negative review:", df[df['sentiment'] == 0]['review'].iloc[0])

    # Clean text data
    print("\nCleaning text data...")
    df['review'] = df['review'].apply(clean_text)

    # Remove stopwords and tokenize
    print("Removing stopwords and tokenizing...")
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda text: tokenize_and_remove_stopwords(text, stop_words))

    # Display examples of processed reviews
    print("\nExamples of processed reviews:")
    print("Positive review:", df[df['sentiment'] == 1]['review'].iloc[0])
    print("Negative review:", df[df['sentiment'] == 0]['review'].iloc[0])

    # Split data into training and testing sets
    print("\nSplitting dataset into train and test sets...")
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=24)

    print("Preprocessing complete.")
    return X_train, X_test, y_train, y_test

# Usage example
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

if path is None:
    raise ValueError("Failed to download dataset. Ensure `kagglehub` is correctly configured.")
dataset_path = f"{path}/IMDB Dataset.csv" # Define dataset path

# Preprocess the IMDB dataset
X_train, X_test, y_train, y_test = preprocess_imdb_data(dataset_path)

# Print the sizes of the training and testing datasets
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
