from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def LSTMModel():
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