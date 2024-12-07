from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel():
    def __init__(self, input_dim, hidden_units=64):
        self.model=None
        self.input_dim = input_dim
        self.hidden_units = hidden_units

    def build_model(self,):
        self.model = Sequential()

        self.model.add(LSTM(self.hidden_units, input_shape=(1, self.input_dim)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model
    
    def train_model(self, X_train, y_train, epochs=20, batch_size=1):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    def predict(self, input):        
        return self.model.predict(input) 
    
    # def save_model(self, filepath):
    #     self.model.save(filepath)

class Trainer():
    def __init__(self,model):
        self.model = model
    
    def save_model(self, location):
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
        self.model.set_params(kernel = kernel, C = C)

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
        self.model.set_params(n_neighbours = n_neighbours, weights = weights)

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
        # self.model.export(location)
        self.model.model.save(f'{location}.h5')

    def validation(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        return accuracy_score(y_val, y_pred)


def save_test_data(X_test, y_test):

    with open('X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)

    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)


def load_data():
    return None

def preprocess(data):
    return None


def main():
    #Preprocess logic
    data = load_data()
    X_train, X_test, X_val,  y_train, y_val, y_test = preprocess(data)
    # change as needed

    save_test_data(X_test, y_test)
    
    input_dim = 200

    my_svm = SVC()
    my_knn = KNeighborsClassifier()
    my_lstm = LSTMModel(input_dim)

    trainers = {
        'SVM' : SVMTrainer(my_svm),
        'KNN' : KNNTrainer(my_knn, n_neighbours=5, weights='uniform'), #change as needed
        'LSTM' : LSTMTrainer(my_lstm),
    }

    
    for trainer_name, trainer in trainers.items():
        print(f'Training: {trainer_name} ...')
        
        trainer.train(X_train, y_train)
        accuracy = trainer.validate(X_val, y_val)

        trainer.save_model(f'./models/{trainer_name}')

        print(f'{trainer_name} acc: {accuracy}')
