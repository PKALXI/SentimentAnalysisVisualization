from abc import ABC, abstractmethod
import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
import pandas as pd
import re
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold

#https://www.tensorflow.org/text/tutorials/text_classification_rnn

class LSTMModel():
    def __init__(self, input_dim=None, hidden_units=64):
        self.model=None
        self.input_dim = input_dim
        self.hidden_units = hidden_units

        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=1)


    def build_model(self,):
        self.model = Sequential()

        self.model.add(LSTM(self.hidden_units, input_shape=(None, self.input_dim)))
        self.model.add(Dense(self.hidden_units, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # opt = tf.keras.optimizers.SGD(learning_rate=0.01) 

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model
    
    def train_model(self, X_train_wrong_shape, y_train, val_size = 0.2, epochs=20, batch_size=1):
        X_train = X_train_wrong_shape.reshape(-1, 1, X_train_wrong_shape.shape[1])

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=val_size, callbacks=[self.tensorboard_callback])
    
    def predict(self, input):
        X_test = input.reshape(-1, 1, input.shape[1])
        y_pred = self.model.predict(X_test) 

        return (y_pred > 0.5).astype(float)
    
    def save_model(self, filepath):
        self.model.save(f'{filepath}.keras')

    def load_model(self, filepath):
        self.model = load_model(filepath)

    def get_params(self):
        return self.model.summary()
        
class Trainer():
    def __init__(self,model):
        self.model = model
    
    def save_model(self, location):
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(location), exist_ok=True)
        
        with open(f'{location}.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def get_params(self):
        return self.model.get_params()

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def validation(self, X_val, y_val):
        pass

class SVMTrainer(Trainer):
    def __init__(self, model:SVC, kernel='rbf', C=1.0):
        super().__init__(model)
        self.model = model
        self.model.set_params(
            kernel=kernel, 
            C=C,
            verbose=True
        )

    def train(self, features, labels):
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
        self.model.fit(X_train, y_train)  

        y_pred = self.model.predict(X_val)

        return accuracy_score(y_val, y_pred)
    
        
    def validation(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        return accuracy_score(y_val, y_pred)
      
class KNNTrainer(Trainer):
    def __init__(self, model:KNeighborsClassifier, n_neighbours, weights):
        super().__init__(model)
        self.model = model
        self.model.set_params(
            n_neighbors = n_neighbours, 
            weights = weights, 
        )

    def train(self, features, labels):
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
        self.model.fit(X_train, y_train)  

        y_pred = self.model.predict(X_val)

        return accuracy_score(y_val, y_pred)
    
    def validation(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        return accuracy_score(y_val, y_pred)

    
class LSTMTrainer(Trainer):
    def __init__(self, model:LSTMModel, epochs=15, batch_size = 32) -> None:
        super().__init__(model)
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train):
        self.model.build_model()
        self.model.train_model(X_train, y_train, val_size = 0.2, batch_size = self.batch_size, epochs = self.epochs)
    
    def save_model(self, location):
        self.model.save_model(location)

    def validation(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        y_pred = y_pred.flatten()
        return accuracy_score(y_val, y_pred)

def save_train_data(X_train, y_train):
    with open('X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)

    with open('y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)

def save_test_data(X_test, y_test):
    with open('X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)

    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

def load_train_data():
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
        
    with open('y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
        
    return X_train, y_train

def load_test_data():
    with open('X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
        
    with open('y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
        
    return X_test, y_test

def store_some_object(obj, path):
    print('Storing: ', path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

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
    
    # print("Dataset loaded successfully. Sample rows:")
    # print(df.head()) # Display first few rows

    # Standardize column names
    if 'sentiment' not in df.columns:
        df.columns = ['review', 'sentiment']

    # Remove duplicates and nulls
    # print("Removing duplicates and null values...")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Convert sentiment labels to binary (0 = negative, 1 = positive)
    # print("Encoding labels...")
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

    # Display examples of unprocessed reviews
    # print("\nExamples of unprocessed reviews:")
    # print("Positive review:", df[df['sentiment'] == 1]['review'].iloc[0])
    # print("Negative review:", df[df['sentiment'] == 0]['review'].iloc[0])

    # Clean text data
    # print("\nCleaning text data...")
    df['review'] = df['review'].apply(clean_text)

    # Remove stopwords and tokenize
    # print("Removing stopwords and tokenizing...")
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda text: tokenize_and_remove_stopwords(text, stop_words))

    # Display examples of processed reviews
    # print("\nExamples of processed reviews:")
    # print("Positive review:", df[df['sentiment'] == 1]['review'].iloc[0])
    # print("Negative review:", df[df['sentiment'] == 0]['review'].iloc[0])

    # Split data into training and testing sets
    # print("\nSplitting dataset into train and test sets...")
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=24)

    print("Preprocessing complete.")
    return X_train, X_test, y_train, y_test

def main():
    LOAD = False

    if not LOAD:
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        dataset_path = f"{path}/IMDB Dataset.csv"

        X_train, X_test, y_train, y_test = preprocess_imdb_data(dataset_path)

        tf_transformer = TfidfVectorizer(
                min_df=2,          # Appear in at least 2 documents
                max_df=0.95,       # Ignore terms that appear in >95% of docs
                use_idf=True,
                ngram_range=(1,3)
            )
        
        print('TF Transform....')
        tf_transformer.fit(X_train, y_train)
        X_train_tf = tf_transformer.transform(X_train)

        store_some_object(tf_transformer, "./transformers/tf_trans.pkl")

        X_test_tf = tf_transformer.transform(X_test)

        print('Reducing Dimensions using PCA')
        pca = PCA(n_components = 200)

        X_train_pca = pca.fit_transform(X_train_tf)

        store_some_object(pca, "./transformers/pca.pkl")

        X_test_pca = pca.transform(X_test_tf)

        save_test_data(X_test_pca, y_test)
        save_train_data(X_train_pca, y_train)
    else:
        X_train_pca, y_train = load_train_data()
        X_test_pca, y_test = load_test_data()

    input_dim = X_train_pca.shape[1]
    my_lstm = LSTMModel(input_dim)

    my_svm = SVC()
    my_knn = KNeighborsClassifier()

    trainers = {
        'SVM' : SVMTrainer(my_svm, kernel='rbf'),
        'KNN' : KNNTrainer(my_knn, n_neighbours=5, weights='uniform'),
        'LSTM' : LSTMTrainer(my_lstm),
    }

    for trainer_name, trainer in trainers.items():
        print(f'Training: {trainer_name} ...')

        validation_acc = trainer.train(X_train_pca, y_train)
        print(f'Validation Accuracy: {validation_acc}')

        # print(f'Validation: {trainer_name} ...')
        # accuracy = trainer.validation(X_test_pca, y_test)
        # print(f'{trainer_name} acc: {accuracy}')

        print(f'Model: {trainer_name} Configuration: ')
        print(trainer.get_params())

        trainer.save_model(f'./models/{trainer_name}')

if __name__ == '__main__':
    main()