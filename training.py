from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from models import LSTMModel
import pickle

model = SVC(kernel = 'rbf', C = 1)

class Trainer():
    def __init__(self,model):
        self.model = model
    
    def save_model(self, location):
        with open(f'{location}.pkl', 'wb') as f:
            pickle.dump(self.model, f)

class SVMTrainer(Trainer):
    def __init__(self, model:SVC, kernel='rbf', C=1.0):
        super().__init__(model)
        self.model = model
        self.model.set_params(kernel = kernel, C = C)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    
class KNNTrainer(Trainer):
    def __init__(self, model:KNeighborsClassifier, n_neighbours, weights):
        super().__init__(model)
        self.model = model
        self.model.set_params(n_neighbours = n_neighbours, weights = weights)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    
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
        self.model.export(location)