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
from tensorflow.keras.optimizers import Adam
import subprocess
import h5py

custom_optimizer = Adam(learning_rate=0.0005)
batch_size = 32
epochs = 15
aantal_candlesticks = 40

mappen = ['trainingfolder', 'AAPL']
bestandsnaam = 'training_data.h5'
bestandsnaam2 = 'val_data.h5'

Xpadnaam = os.path.join(*mappen, bestandsnaam)
ypadnaam = os.path.join(*mappen, bestandsnaam2)

with h5py.File(Xpadnaam, "r") as file:
    # Haal de datasets uit het bestand en laad ze in variabelen
    Xtrain = file["Xtrain"][:]
    ytrain = file["ytrain"][:]

with h5py.File("val_data.h5", "r") as file:
    # Haal de datasets uit het bestand en laad ze in variabelen
    Xval = file["Xval"][:]
    yval = file["yval"][:]

Xmax = Xtrain.max()
ymax = ytrain.max()

Xtrain = Xtrain / Xmax
ytrain = ytrain / ymax

Xvmax = Xval.max()
yvmax = ymax.max()

Xval = Xval / Xvmax
yval = yval / yvmax
input_shape = (Xtrain.shape[1], Xtrain.shape[2])
output_shape = ytrain.shape[1]

def get_training(input_shape=input_shape):
	model = laden_of_maken(input_shape)
	mappen = os.listdir("trainingfolder")
	random.shuffle(mappen)
	for mapp in mappen:
		Xpadnaam = os.path.join("trainingfolder", mapp, "training_data.h5")
		print("Currently training on:",  mapp)
		with h5py.File(Xpadnaam, "r") as file:
		    # Haal de datasets uit het bestand en laad ze in variabelen
		    Xtrain = file["Xtrain"][:]
		    ytrain = file["ytrain"][:]

		Xmax = Xtrain.max()
		ymax = ytrain.max()

		Xtrain = Xtrain / Xmax
		ytrain = ytrain / ymax

		training(model, Xtrain, ytrain)
		sla_model_op(model, "model")

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

def training(model, Xtrain, ytrain):
	model.compile(loss='mean_absolute_percentage_error', optimizer=custom_optimizer)
	model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size)

def evalueer_model(model, X, y, Xmax=Xvmax, ymax=yvmax):
    voorspellingen = model.predict(X)
    voorspellingen *= Xvmax
    y *= yvmax
    mae = mean_absolute_error(y, voorspellingen)
    mse = mean_squared_error(y, voorspellingen)
    return mae, mse, voorspellingen

get_training()
model = load_model("model.keras")
mae, mse, voorspellingen = evalueer_model(model, Xval, yval)
print("MAE: ", {mae})
print("MSE: ", {mse})