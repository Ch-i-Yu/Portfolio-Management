# import necessary dependencies

from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)
simplefilter(action = "ignore", category = DeprecationWarning)

import gc
import os
import re

import time

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
    """
    Predict Future Stock Prices using Recurrent Neural Network with Long Short-Term Memory.
    Designed for Debugging Scenarios instead of Deployment Scenarios.
    """

    def __init__(self, stockCode: str,
                       LOOKBACK: int,
                       BATCH_SIZE: int,
                       EPOCHS: int,
                       verbose = False):
        # Set Stock Code
        self.stockCode = stockCode

        # Set Hyper Parameters
        self.LOOKBACK = LOOKBACK
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS

        # Initialize Data
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []

        # Config Utilities Variables
        self.verbose = verbose
        self.model = None
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        self.localtime = time.strftime(r"%m-%d_%H%M" ,time.localtime(time.time()))



    def loadStock(self):
        df = pd.read_csv(r".StockDatasets/{}.csv".format(self.stockCode), index_col = "Date")
        df.index = pd.DatetimeIndex(self.df.index)

        if self.verbose:
            pass

        return df

    def clearStock(df: pd.DataFrame):
        # feature expansion: percent changes
        df["Pct Change"] = df["Adj Close"].pct_change()
        df["Pct Change"][np.isnan(df["Pct Change"])] = 0

        # delete unapplied columns
        del df["Open"]
        del df["High"]
        del df["Low"]
        del df["Close"]
        del df["Volume"]
        gc.collect()

        return df
    
    def splitTrainTest(df: pd.DataFrame):
        train = df[:"2018"]    # 2011-1-1 ~ 2018-12-31
        test = df[:"2018"]    # 2019-1-1 ~ 2021-12-31

        return train, test

    def timestampStock(self, train: pd.DataFrame,
                             test: pd.DataFrame):

        trainBOUND = train.shape[0]
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


    def activateModel(self):
        # Create model
        self.model = Sequential()

        # Input Layers
        self.model.add(LSTM(
                       units = 64,
                       return_sequences = True,
                       input_shape = (self.X_train.shape[1], 1)),
                       dropout = 0.1)

        # Hidden Layers
        self.model.add(LSTM(
                       units = 32,
                       return_sequences = True),
                       dropout = 0.1)

        self.model.add(LSTM(
                       units = 16,
                       return_sequences = True),
                       dropout = 0.1)

        self.model.add(LSTM(
                       units = 8,
                       return_sequences = True),
                       dropout = 0.1)

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
                                      batch_size= self.BATCH_SIZE,
                                      epochs = self.EPOCHS,
                                      callbacks = [stop],
                                      validation_split = 0.2)

        # Save Model
        self.model.save("./Models/{}-{}.tf".format(self.localtime,self.stockCode),
                        save_format = "tf")

        # Plot Accuracy & Loss
        if self.verbose:
            plt.figure()

            for key in self.history.history.keys():
                plt.plot(np.arange(0, self.EPOCHS) + 1,
                                self.history.history[key],
                                label = key)
            
            plt.suptitle("Training Loss & Accuracy - {}".format(self.stockCode))
            plt.xlabel("Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")

            plt.savefig("./Plots/{}-{}-Loss_Accuracy.png".format(self.localtime,self.stockCode))
        
    def predictStockPrice(self):
        # Apply Feature Scaling to Test Data
        self.X_test = self.scaler.transform(self.X_test)

        predicted_StockPrice = self.model.predict(self.X_test)
        self.Y_pred = self.scaler.inverse_transform(predicted_StockPrice)

        # Plot Predictions
        if self.verbose:
            plt.plot(self.Y_test, color = "red",
                    label = "Real {} Stock Price".format(self.stockCode))

            plt.plot(self.Y_pred, color = "blue",
                    label = "Predicted {} Stock Price".format(self.stockCode))

            plt.suptitle("{} Stock Price Prediction".format(self.stockCode))
            plt.title("RMS Error: {}".format(np.square(self.Y_test - self.Y_pred).mean(axis = None)))
            plt.xlabel("Dates")
            plt.ylabel("Prices")
            plt.legend(loc="lower left")

            plt.savefig("./Plots/{}-{}-Loss_Accuracy.png".format(self.localtime,self.stockCode))