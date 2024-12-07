from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from models import LSTMModel

model = SVC(kernel = 'rbf', C = 1)


class SVMTrainer():
    def __init__(self, model:SVC, kernel='rbf', C=1.0):
        self.model = model
        self.model.set_params(kernel = kernel, C = C)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    def validate(self, X_val, y_val):
        y_pred = self.model.predict(X_val)

        return accuracy_score(y_val, y_pred)
    
class KNNTrainer():
    def __init__(self, model:KNeighborsClassifier, n_neighbours, weights):
        self.model = model
        self.model.set_params(n_neighbours = n_neighbours, weights = weights)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    def validate(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        return accuracy_score(y_val, y_pred) 
    
class LSTMTrainer():
    def __init__(self, model:LSTMModel) -> None:
        self.model = model

    def train():
        self.model.build_model()
        