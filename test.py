import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from abc import ABC, abstractmethod
from training import LSTMModel

class Evaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def evaluate(self, y_test, y_pred):
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test data: {accuracy:.4f}")

        precision = precision_score(y_test, y_pred)
        print(f"Precision on test data: {precision:.4f}")

        recall = recall_score(y_test, y_pred)
        print(f"Recall on test data: {recall:.4f}")


class SKLearnEvaluator(Evaluator):
    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X_test):
        return  self.model.predict(X_test)

        
class LSTMEvaluator(Evaluator):
    def load_model(self):
        self.model = LSTMModel()
        self.model.load_model(self.model_path)

    def predict(self, X_test):
        return  self.model.predict(X_test)

def load_test_data():
    with open('X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    return X_test, y_test


def main():
    print("Loading test data...")
    X_test, y_test = load_test_data()

    models = {
        "SVM": ("./models/SVM.pkl", SKLearnEvaluator),
        "KNN": ("./models/KNN.pkl", SKLearnEvaluator),
        "LSTM": ("./models/LSTM.keras", LSTMEvaluator)  # Update path
    }

    for model_name, (path, evaluator_class) in models.items():
        print(f"\nEvaluating {model_name} model...")
        evaluator = evaluator_class(path)
        evaluator.load_model()
        y_pred = evaluator.predict(X_test)
        evaluator.evaluate(y_test, y_pred)


if __name__ == "__main__":
    main()
