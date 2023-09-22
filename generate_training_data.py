import pandas as pd
import numpy as np
import math
import random

gewenste_aantal_jaren = 7
aantal_candlesticks = 30
aantal_sets = 3200

def rijen_knippen(gewenste_aantal_jaren):
	data = pd.read_csv("spx_data.csv")
	totaal_aantal_jaren_in_dataset = len(data)
	aantal_rijen_te_behouden = math.floor((len(data)) / 20 * gewenste_aantal_jaren)
	data = data[-aantal_rijen_te_behouden:]
	return data

def maak_training_set(aantal_sets, aantal_candlesticks):
	data = rijen_knippen(gewenste_aantal_jaren)
	max_startpunt = len(data) - (aantal_candlesticks+1)
	X, y = [], []
	for i in range(aantal_sets):
		startpunt = random.randint(0, max_startpunt)
		X.append(data.iloc[startpunt:startpunt+aantal_candlesticks].values)
		y.append(data.iloc[startpunt+aantal_candlesticks][["Open", "Close"]].values)
	X = np.array(X)
	y = np.array(y)
	return X, y

Xtrain, ytrain = maak_training_set(aantal_sets, aantal_candlesticks)
combined_data = pd.DataFrame(np.hstack((Xtrain.reshape(Xtrain.shape[0], -1), ytrain)))
combined_data.to_csv("training_data.csv", index=False, mode="w")
Xtrain, ytrain = maak_training_set(aantal_sets, aantal_candlesticks)
combined_data = pd.DataFrame(np.hstack((Xtrain.reshape(Xtrain.shape[0], -1), ytrain)))
combined_data.to_csv("val_data.csv", index=False, mode="w")
