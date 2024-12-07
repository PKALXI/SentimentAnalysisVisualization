from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import kagglehub
from preprocess import preprocess_imdb_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from tensorflow.keras.models import load_model

#https://www.tensorflow.org/text/tutorials/text_classification_rnn

class LSTMModel():
    def __init__(self, input_dim=None, hidden_units=64):
        self.model=None
        self.input_dim = input_dim
        self.hidden_units = hidden_units

    def build_model(self,):
        self.model = Sequential()

        self.model.add(LSTM(self.hidden_units, input_shape=(None, self.input_dim)))
        self.model.add(Dense(self.hidden_units, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model
    
    def train_model(self, X_train_wrong_shape, y_train, epochs=20, batch_size=1):
        X_train = X_train_wrong_shape.reshape(-1, 1, X_train_wrong_shape.shape[1])

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    def predict(self, input):
        X_test = input.reshape(-1, 1, input.shape[1])
        y_pred = self.model.predict(X_test) 

        return (y_pred > 0.5).astype(float)
    
    def save_model(self, filepath):
        self.model.save(f'{filepath}.keras')

    def load_model(self, filepath):
        self.model = load_model(filepath)
        

class Trainer():
    def __init__(self,model):
        self.model = model
    
    def save_model(self, location):
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(location), exist_ok=True)
        
        with open(f'{location}.pkl', 'wb') as f:
            pickle.dump(self.model, f)

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
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    def validation(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        return accuracy_score(y_val, y_pred)
    
    
class KNNTrainer(Trainer):
    def __init__(self, model:KNeighborsClassifier, n_neighbours, weights):
        super().__init__(model)
        self.model = model
        self.model.set_params(n_neighbors = n_neighbours, weights = weights)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
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
        self.model.train_model(X_train, y_train, batch_size = self.batch_size, epochs = self.epochs)
    
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


def main():
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
    X_test_tf = tf_transformer.transform(X_test)

    print('PCA dimensions')
    pca = PCA(n_components = 200)

    X_train_pca = pca.fit_transform(X_train_tf)
    X_test_pca = pca.transform(X_test_tf)

    save_test_data(X_test_pca, y_test)
    save_train_data(X_train_pca, y_train)

    X_train, y_train = load_train_data()
    X_val, y_val = load_test_data()

    # X_train_pca, y_train = load_train_data()
    # X_test_pca, y_test = load_test_data()

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
        trainer.train(X_train_pca, y_train)

        print(f'Validation: {trainer_name} ...')
        accuracy = trainer.validation(X_train_pca, y_test)
        print(f'{trainer_name} acc: {accuracy}')


        trainer.save_model(f'./models/{trainer_name}')

if __name__ == '__main__':
    main()