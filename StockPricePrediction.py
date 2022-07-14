# import necessary dependencies

from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)
simplefilter(action = "ignore", category = DeprecationWarning)

import gc
import os
import re

import numpy as np
import pandas as pd
np.random.seed(42)

import matplotlib.pyplot as plt
import seaborn as sns

sns.color_palette("rocket_r", as_cmap=True)
sns.set_style("white")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU


class StockPricePrediction:
    """Predict Future Stock Prices using Recurrent Neural Network with Long Short-Term Memory."""
    def _init_(self):
        self.args = None

    def _init_(self, args):
        self.args = args