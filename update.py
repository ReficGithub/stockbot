import os
import math
import random
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
import plotly.graph_objects as go
import pytz

tradingdays = 75
aanta_candlesticks = 10
huidige_datum = date.today()
einddatum = huidige_datum + timedelta(days=1)
startdatum = huidige_datum - timedelta(days=7000)

def finance_ophalen(startdatum=startdatum, einddatum=einddatum):
    ticker = "^SPX"
    data = yf.download(ticker, start=startdatum, end=einddatum)
    data.drop("Adj Close", axis=1, inplace=True)
    data.drop("Volume", axis=1, inplace=True)
    data = np.array(data)
    return data

def mappen_ophalen():
    huidige_directory = os.getcwd()
    mappen = []
    for item in os.listdir(huidige_directory):
        if os.path.isdir(item):
            mappen.append(item)
    return mappen

def voorspellen(tradingdays):
    aantal_dagen = tradingdays
    data = finance_ophalen()
    Xmax = data.max()
    data = data / Xmax
    voorspel_reeks = []
    for i in range(aantal_dagen):
        data = data[:-1]
        reeks = data[-10:]
        voorspel_reeks.append(reeks)
    aantal_modellen = mappen_ophalen()
    lijst_voorspellingen = []
    for mapp in aantal_modellen:
        mapnaam = mapp
        model_pad = os.path.join(mapnaam, "model.keras")
        model = load_model(model_pad)
        voorspel_reeks = np.array(voorspel_reeks)
        voorspellingen = model.predict(voorspel_reeks)
        voorspellingen = np.array(voorspellingen)
        voorspellingen = np.flip(voorspellingen, axis=0)
        # print("Model", mapnaam, "\n", voorspellingen)
        lijst_voorspellingen.append([mapnaam, voorspellingen])
    return lijst_voorspellingen, Xmax
    

lijst_voorspellingen, Xmax = voorspellen(tradingdays)
lijst_voorspellingen[0][1] *= Xmax
echte_prijzen = []
data = finance_ophalen()
data = data[-tradingdays:]

# Lijsten om de kaarsgegevens op te slaan
open_prijzen = []
high_prijzen = []
low_prijzen = []
close_prijzen = []

namen = []

# Voor elke set gegevens
for naam, tijdreeks in lijst_voorspellingen:
    open_prijzen.append(tijdreeks[:, 0])
    high_prijzen.append(tijdreeks[:, 1])
    low_prijzen.append(tijdreeks[:, 2])
    close_prijzen.append(tijdreeks[:, 3])
    namen.append(naam)

open_prijzen.append(data[:, 0])
high_prijzen.append(data[:, 1])
low_prijzen.append(data[:, 2])
close_prijzen.append(data[:, 3])
namen.append('SPX')

increasing_color = 'lightblue'
decreasing_color = 'pink'

fig = go.Figure()

for i in range(len(namen)):
    if namen[i] != 'SPX':
        fig.add_trace(go.Candlestick(
            x=np.arange(len(open_prijzen[i])),
            open=open_prijzen[i],
            high=high_prijzen[i],
            low=low_prijzen[i],
            close=close_prijzen[i],
            name=namen[i],
            increasing_line_color=increasing_color,
            decreasing_line_color=decreasing_color
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=np.arange(len(open_prijzen[i])),
            open=open_prijzen[i],
            high=high_prijzen[i],
            low=low_prijzen[i],
            close=close_prijzen[i],
            name=namen[i]
        ))

# Voeg een titel toe aan de grafiek
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(title="Real price of SPX versus NN's")

# Toon de grafiek
fig.show()