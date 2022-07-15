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
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import rmsprop_v2
from keras import callbacks


class StockPricePrediction:
    """Predict Future Stock Prices using Recurrent Neural Network with Long Short-Term Memory."""
    
    # Class Variables:

    # HyperParameters: EPOCHS, BATCHSIZE, LOOKBACK
    # Class Attributes(Manually Allocated): verbose, stockCode, scaler
    # Class Attributes(Automatic Allocated): df, X_train, Y_train


    def __init__(self, args, stockCode, LOOKBACK):
        self.args = args

        self.stockCode = stockCode

        self.LOOKBACK = LOOKBACK

        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []

        self.model = None
        self.scaler = MinMaxScaler(feature_range = (0, 1))



    def loadStock(self, verbose = False):
        self.df = pd.read_csv(r".StockDatasets/{}.csv".format(self.stockCode), index_col = "Date")
        self.df.index = pd.DatetimeIndex(self.df.index)

        if verbose:
            pass


    def preprocessStock(self, verbose = False):
        self.loadStock()

        # feature expansion: percent changes
        self.df["Pct Change"] = self.df["Adj Close"].pct_change()
        self.df["Pct Change"][np.isnan(self.df["Pct Change"])] = 0

        # delete unapplied columns
        del self.df["Open"]
        del self.df["High"]
        del self.df["Low"]
        del self.df["Close"]
        del self.df["Volume"]
        gc.collect()


    def timestampStock(self, verbose = False):
        # Train-Test Dataset Split
        self.preprocessStock()
        train = self.df[:"2018"]    # 2011-1-1 ~ 2018-12-31
        trainBOUND = train.shape[0]

        test = self.df[:"2018"]    # 2019-1-1 ~ 2021-12-31
        testBound = test.shape[0]

        # Apply Feature Scaling to Train Data
        train_scaled = self.scaler.fit_transform(train)
        del train
        gc.collect()

        # Create Time Stamp
        for i in range(self.LOOKBACK, trainBOUND - 1):
            self.X_train.append(train_scaled[i - self.LOOKBACK : i])
            self.Y_train.append(train_scaled[i]["Adj Close"])

        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)

        for i in range(self.loadStock, testBound - 1):
            self.X_test.append(test[i - self.LOOKBACK : i])
            self.Y_test.append(test[i]["Adj Close"])

        self.X_test = np.array(self.X_test)

        # Reshape Inputs
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))


    def compileModel(self):
        # Create model
        self.model = Sequential()

        # Input Layers
        self.model.add(LSTM(
                       units = 64,
                       return_sequences = True,
                       input_shape = (self.X_train.shape[1], 1)))
        self.model.add(Dropout(0.25))

        # Hidden Layers
        self.model.add(LSTM(
                       units = 32,
                       return_sequences = True))
        self.model.add(Dropout(0.25))

        self.model.add(LSTM(
                       units = 16,
                       return_sequences = True))
        self.model.add(Dropout(0.25))

        self.model.add(LSTM(
                       units = 8,
                       return_sequences = True))
        self.model.add(Dropout(0.25))

        # Output Layer
        self.model.add(Dense(units = 1))

        # Set Optimizer
        opt = rmsprop_v2.RMSprop(learning_rate = 10e-6)

        # Set Callbacks: Early Stopping
        stop = callbacks.EarlyStopping(
            min_delta = 10e-4,
            patience = 16,
            restore_best_weights = True
        )

        # Compile Model
        self.model.compile(optimizer = opt,
                           loss = "mean_squared_error",
                           metrics = ["mean_squared_error"])

        # Train Model
        self.history = self.model.fit(self.X_train, self.Y_train,
                                      batch_size= self.BATCHSIZE,
                                      epochs = self.EPOCHS,
                                      callbacks = [stop],
                                      validation_split = 0.2)

        
def predictStockPrice(self):
    # Apply Feature Scaling to Test Data
    self.X_test = self.scaler.transform(self.X_test)

    predicted_StockPrice = self.model.predict(self.X_test)
    self.Y_pred = self.scaler.inverse_transform(predicted_StockPrice)