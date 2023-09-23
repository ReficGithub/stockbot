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
import subprocess

combined_data = pd.read_csv("training_data.csv")
val_combined = pd.read_csv("val_data.csv")

aantal_candlesticks = 10

Xtrain = combined_data.iloc[:, :-2].values
ytrain = combined_data.iloc[:, -2:].values
Xtrain = Xtrain[:, :aantal_candlesticks * 4].reshape(-1, aantal_candlesticks, 4)

Xval = val_combined.iloc[:, :-2].values
yval = val_combined.iloc[:, -2:].values
Xval = Xval[:, :aantal_candlesticks * 4].reshape(-1, aantal_candlesticks, 4)

Xmax = Xtrain.max()
ymax = ytrain.max()

Xtrain = Xtrain / Xmax
ytrain = ytrain / ymax

input_shape = (Xtrain.shape[1], Xtrain.shape[2])
output_shape = ytrain.shape[1]
batch_size = 32
epochs = 25



def bouw_lstm_netwerk(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(1250, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(1000, return_sequences=True))
    model.add(LSTM(1250, return_sequences=True))
    model.add(LSTM(750))   
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

def evalueer_model(model, X, y, Xmax=Xmax, ymax=ymax):
    voorspellingen = model.predict(X)
    mae = mean_absolute_error(y, voorspellingen)
    mse = mean_squared_error(y, voorspellingen)
    voorspellingen*=Xmax
    y*=ymax
    return mae, mse, voorspellingen


model = laden_of_maken(input_shape)
training(model)
sla_model_op(model, "model")

mae, mse, voorspellingen = evalueer_model(model, Xval, yval)
print("MAE: ", {mae})
print("MSE: ", {mse})