# coding: iso-8859-1 -*-
import pandas as pd
import numpy as np
import math
import random
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
#pip install pandas numpy math random datetime scikit-learn

combined_data = pd.read_csv("training_data.csv")
val_combined = pd.read_csv("val_data.csv")

Xtrain = combined_data.iloc[:, :-2].values
ytrain = combined_data.iloc[:, -2:].values
Xtrain = Xtrain[:, :30 * 6].reshape(-1, 30, 6)

Xval = val_combined.iloc[:, :-2].values
yval = val_combined.iloc[:, -2:].values
Xval = Xval[:, :30 * 6].reshape(-1, 30, 6)

input_shape = (Xtrain.shape[1], Xtrain.shape[2])
output_shape = ytrain.shape[1]
batch_size = 32
epochs = 50

def bouw_lstm_netwerk(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(1250, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(1250, return_sequences=True))
    model.add(LSTM(500, return_sequences=True))
    model.add(LSTM(500))   
    model.add(Dense(output_shape))
    return model

def sla_model_op(model, model_naam):
    model.save(f'{model_naam}.keras')
    print(f'{model_naam}.keras is succesvol opgeslagen.')

def laden_of_maken(input_shape):
	if os.path.isfile("model.keras"):
		model = load_model("model.keras")
		print("Loaded model.")
	else:
		model = bouw_lstm_netwerk(input_shape, output_shape)
		print("Creating model.")
	return model

def training(model):
	model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
	model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size)

def evalueer_model(model, X, y):
    voorspellingen = model.predict(X)
    voorspellingen*=10000
    y*=10000
    mae = mean_absolute_error(y, voorspellingen)
    mse = mean_squared_error(y, voorspellingen)
    return mae, mse, voorspellingen


model = laden_of_maken(input_shape)
training(model)
sla_model_op(model, "model")

mae, mse, voorspellingen = evalueer_model(model, Xval, yval)
print("MAE: ", {mae})
print("MSE: ", {mse})
