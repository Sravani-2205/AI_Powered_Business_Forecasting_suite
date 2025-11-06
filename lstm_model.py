# lstm_model.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging

def run_lstm(df, window_size=3):
    try:
        data = df['Sales'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(scaled) - window_size):
            X.append(scaled[i:i+window_size])
            y.append(scaled[i+window_size])
        X, y = np.array(X), np.array(y)

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=100, verbose=0)

        last_seq = scaled[-window_size:].reshape((1, window_size, 1))
        prediction = model.predict(last_seq)
        logging.info("LSTM model forecast generated.")
        return scaler.inverse_transform(prediction)[0][0]
    except Exception as e:
        logging.error(f"LSTM model error: {e}")
        raise